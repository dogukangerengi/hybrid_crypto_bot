# =============================================================================
# AI ENTRY OPTİMİZER (GEMİNİ API)
# =============================================================================
# Amaç: IC analiz sonuçlarını, risk hesaplamalarını ve market verisini
#        Gemini API'ye göndererek nihai işlem kararı almak.
#
# Pipeline:
# 1. IC analiz sonuçları (hangi TF, hangi indikatörler, yön, güven)
# 2. Risk hesaplamaları (SL/TP, pozisyon büyüklüğü, margin)
# 3. Market context (fiyat, volatilite, rejim, trend)
# 4. → Gemini structured prompt → JSON response parse
# 5. → GateKeeper entegrasyonu (IC skoru < 55 → NO_TRADE)
# 6. → Nihai karar: LONG / SHORT / WAIT
#
# Gate Keeper Eşikleri (config.py):
# - < 55  → NO_TRADE (işlem yapma)
# - 55-70 → REPORT_ONLY (sadece bildir, emir gönderme)
# - > 70  → FULL_TRADE (Gemini onaylarsa emir gönder)
#
# Gemini Kullanım Gerekçesi:
# - IC istatistiksel filtre → "hangi indikatörler anlamlı?"
# - Gemini semantik analiz → "bu bağlamda sinyal mantıklı mı?"
# - İnsan trader'ın son kontrol adımını otomatize eder
# - Tek başına karar vermez, IC filtresinden geçenleri değerlendirir
#
# Kullanım:
# --------
# from ai.gemini_optimizer import GeminiOptimizer
# optimizer = GeminiOptimizer()
# decision = optimizer.get_decision(ic_data, risk_data, market_data)
# =============================================================================

import sys                                     # Path ayarları
import json                                    # JSON parse
import time                                    # Rate limiting
import logging                                 # Log yönetimi
from pathlib import Path                       # Platform-bağımsız dosya yolları
from typing import Dict, List, Optional, Any   # Tip belirteçleri
from dataclasses import dataclass, field       # Yapılandırılmış veri sınıfı
from datetime import datetime, timezone        # Zaman damgası
from enum import Enum                          # Sabit değer enumları

# Proje config import
sys.path.insert(0, str(Path(__file__).parent.parent))  # → src/
from config import cfg                         # Merkezi config (AIConfig, GateKeeperConfig dahil)

# Logger
logger = logging.getLogger(__name__)


# =============================================================================
# ENUM & DATACLASS TANIMLARI
# =============================================================================

class AIDecision(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    WAIT = "WAIT"
    
    # ⬇️ BU METODU EKLE ⬇️
    @classmethod
    def from_direction(cls, direction: str) -> 'AIDecisionType':
        d = (direction or "").upper()
        if d in ("LONG", "BUY", "BULLISH"):
            return cls.LONG
        elif d in ("SHORT", "SELL", "BEARISH"):
            return cls.SHORT
        return cls.WAIT

class GateAction(Enum):
    """Gate Keeper aksiyonları."""
    NO_TRADE = "NO_TRADE"                      # IC < 55 → işlem yapma
    REPORT_ONLY = "REPORT_ONLY"                # 55 ≤ IC < 70 → sadece bildir
    FULL_TRADE = "FULL_TRADE"                  # IC ≥ 70 → tam işlem


@dataclass
class AIAnalysisInput:
    """
    Gemini'ye gönderilecek analiz girdisi.
    
    IC analiz, risk hesaplama ve market context'i tek bir
    yapıda toplar. Prompt builder bu objeyi kullanır.
    """
    # Sembol bilgileri
    symbol: str                                # 'BTC/USDT:USDT'
    coin: str                                  # 'BTC'
    price: float                               # Son fiyat ($)
    change_24h: float                          # 24h % değişim
    
    # IC analiz sonuçları
    best_timeframe: str                        # En iyi TF (örn: '1h')
    ic_confidence: float                       # IC güven skoru (0-100)
    ic_direction: str                          # IC'nin önerdiği yön ('LONG'/'SHORT'/'NEUTRAL')
    
    # Kategori bazlı top indikatörler
    category_tops: Dict[str, Dict]             # {'trend': {'name': 'EMA_20', 'ic': 0.15}, ...}
    
    # TF sıralaması
    tf_rankings: List[Dict]                    # [{'tf': '1h', 'score': 75, 'direction': 'SHORT'}, ...]
    
    # Risk hesaplamaları
    atr: float = 0.0                           # ATR değeri ($)
    atr_pct: float = 0.0                       # ATR yüzde
    sl_price: float = 0.0                      # Önerilen SL ($)
    tp_price: float = 0.0                      # Önerilen TP ($)
    risk_reward: float = 0.0                   # RR oranı
    position_size: float = 0.0                 # Pozisyon büyüklüğü
    leverage: int = 0                          # Önerilen kaldıraç
    
    # Market context
    market_regime: str = "unknown"             # 'trending'/'ranging'/'volatile'
    volume_24h: float = 0.0                    # 24h USDT hacim
    volatility: float = 0.0                    # 24h volatilite (%)


@dataclass
class AIDecisionResult:
    """
    Gemini'nin kararı + Gate Keeper sonucu.
    
    Bu obje execution modülüne gönderilir.
    gate_action FULL_TRADE değilse emir gönderilmez.
    """
    # AI kararı
    decision: AIDecision                       # LONG / SHORT / WAIT
    confidence: float                          # AI güven skoru (0-100)
    reasoning: str                             # Karar gerekçesi (Türkçe)
    
    # Gate Keeper
    gate_action: GateAction                    # NO_TRADE / REPORT_ONLY / FULL_TRADE
    ic_score: float                            # IC güven skoru
    
    # Ek bilgiler
    entry_price: float = 0.0                   # Önerilen giriş fiyatı
    sl_price: float = 0.0                      # Önerilen SL
    tp_price: float = 0.0                      # Önerilen TP
    risk_reward: float = 0.0                   # Hesaplanan RR
    atr_multiplier: float = 1.5                # Önerilen ATR çarpanı
    
    # Meta
    model_used: str = ""                       # Hangi Gemini modeli kullanıldı
    timestamp: str = ""                        # Karar zamanı
    raw_response: str = ""                     # Ham Gemini yanıtı (debug)
    
    def should_execute(self) -> bool:
        """İşlem gönderilmeli mi?"""
        return (
            self.gate_action == GateAction.FULL_TRADE
            and self.decision in [AIDecision.LONG, AIDecision.SHORT]
            and self.confidence >= 60
        )
    
    def summary(self) -> str:
        """Telegram mesajı için özet."""
        dec_emoji = {
            AIDecision.LONG: "🟢 LONG",
            AIDecision.SHORT: "🔴 SHORT",
            AIDecision.WAIT: "⏳ BEKLE"
        }
        gate_emoji = {
            GateAction.NO_TRADE: "🚫",
            GateAction.REPORT_ONLY: "📋",
            GateAction.FULL_TRADE: "✅"
        }
        
        lines = [
            f"🤖 AI Karar: {dec_emoji.get(self.decision, '❓')}",
            f"🎯 Güven: {self.confidence:.0f}/100",
            f"📊 IC Skor: {self.ic_score:.0f}/100",
            f"🚦 Gate: {gate_emoji.get(self.gate_action, '❓')} {self.gate_action.value}",
            f"",
            f"💬 {self.reasoning}",
        ]
        
        if self.should_execute():
            lines.extend([
                f"",
                f"📍 Entry: ${self.entry_price:,.2f}",
                f"🛑 SL: ${self.sl_price:,.2f}",
                f"🎯 TP: ${self.tp_price:,.2f}",
                f"⚖️ RR: {self.risk_reward:.1f}",
            ])
        
        return "\n".join(lines)


# =============================================================================
# ANA AI OPTİMİZER SINIFI
# =============================================================================

class GeminiOptimizer:
    """
    Gemini API ile IC analiz sonuçlarını değerlendirip
    nihai LONG/SHORT/WAIT kararı veren AI optimizer.
    
    Flow:
    1. Gate Keeper kontrolü (IC skoru eşik altıysa direkt WAIT)
    2. Structured prompt oluştur
    3. Gemini API çağrısı (fallback modellerle retry)
    4. JSON response parse
    5. AIDecisionResult döndür
    """
    
    def __init__(self):
        """
        GeminiOptimizer başlatır.
        
        google-generativeai kütüphanesini lazy import eder.
        API key config'den (GEMINI_API_KEY) okunur.
        """
        self.ai_cfg = cfg.ai                   # AIConfig: model, temperature, retries
        self.gate_cfg = cfg.gate               # GateKeeperConfig: eşikler
        self._client = None                    # Lazy init (ilk çağrıda oluşur)
        
        logger.info(
            f"GeminiOptimizer başlatıldı | "
            f"Model: {self.ai_cfg.model} | "
            f"API: {'✅' if self.ai_cfg.is_configured() else '❌'}"
        )
    
    # =========================================================================
    # LAZY CLIENT INIT
    # =========================================================================
    
    def _get_client(self):
        """
        Gemini client'ı lazy olarak başlatır.
        
        İlk API çağrısına kadar kütüphane import edilmez.
        Bu sayede Gemini key yokken bile diğer modüller çalışır.
        """
        if self._client is None:
            try:
                import google.generativeai as genai     # Lazy import
                
                if not self.ai_cfg.is_configured():
                    raise ValueError(
                        "GEMINI_API_KEY ayarlanmamış! "
                        ".env dosyasına GEMINI_API_KEY=... ekleyin."
                    )
                
                genai.configure(api_key=self.ai_cfg.api_key)
                self._client = genai
                logger.info("Gemini client başlatıldı ✅")
                
            except ImportError:
                raise ImportError(
                    "google-generativeai yüklü değil! "
                    "pip install google-generativeai"
                )
        
        return self._client
    
    # =========================================================================
    # ANA KARAR FONKSİYONU
    # =========================================================================
    
    def get_decision(
        self,
        analysis_input: AIAnalysisInput
    ) -> AIDecisionResult:
        """
        IC analiz sonuçlarını değerlendirip nihai karar verir.
        
        Pipeline:
        1. Gate Keeper → IC skoru kontrolü
        2. Gemini prompt → API çağrısı
        3. Response parse → AIDecisionResult
        
        Parameters:
        ----------
        analysis_input : AIAnalysisInput
            IC analiz + risk + market verileri
            
        Returns:
        -------
        AIDecisionResult
            Nihai karar + gerekçe + gate action
        """
        timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        
        # ---- GATE KEEPER KONTROLÜ ----
        gate_action = self._check_gate(analysis_input.ic_confidence)
        
        if gate_action == GateAction.NO_TRADE:
            logger.info(
                f"🚫 Gate: NO_TRADE (IC={analysis_input.ic_confidence:.0f} "
                f"< {self.gate_cfg.no_trade})"
            )
            return AIDecisionResult(
                decision=AIDecision.WAIT,
                confidence=0,
                reasoning=f"IC skoru ({analysis_input.ic_confidence:.0f}) "
                          f"eşik altında ({self.gate_cfg.no_trade}). İşlem yok.",
                gate_action=GateAction.NO_TRADE,
                ic_score=analysis_input.ic_confidence,
                model_used="gate_keeper",
                timestamp=timestamp
            )
        
        # ---- GEMİNİ API ÇAĞRISI ----
        try:
            prompt = self._build_prompt(analysis_input)
            raw_response = self._call_gemini(prompt)
            parsed = self._parse_response(raw_response)
            
            # Karar objesini oluştur
            decision = AIDecision[parsed.get('decision', 'WAIT').upper()]
            confidence = float(parsed.get('confidence', 50))
            reasoning = parsed.get('reasoning', 'Gerekçe alınamadı')
            atr_mult = float(parsed.get('atr_multiplier', 1.5))
            
            result = AIDecisionResult(
                decision=decision,
                confidence=confidence,
                reasoning=reasoning,
                gate_action=gate_action,
                ic_score=analysis_input.ic_confidence,
                entry_price=analysis_input.price,
                sl_price=analysis_input.sl_price,
                tp_price=analysis_input.tp_price,
                risk_reward=analysis_input.risk_reward,
                atr_multiplier=atr_mult,
                model_used=self.ai_cfg.model,
                timestamp=timestamp,
                raw_response=raw_response[:500]        # Debug için ilk 500 char
            )
            
            logger.info(
                f"🤖 AI Karar: {decision.value} | "
                f"Güven: {confidence:.0f} | "
                f"Gate: {gate_action.value}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Gemini hatası: {e}")
            
            # Fallback: IC yönünü kullan ama düşük güvenle
            fallback_decision = self._ic_fallback(analysis_input)
            fallback_decision.gate_action = gate_action
            fallback_decision.timestamp = timestamp
            return fallback_decision
    
    # =========================================================================
    # GATE KEEPER
    # =========================================================================
    
    def _check_gate(self, ic_score: float) -> GateAction:
        """
        IC skoru eşiklerine göre gate action belirler.
        
        Eşikler (config.py → GateKeeperConfig):
        - < 55  → NO_TRADE: İstatistiksel sinyal yetersiz
        - 55-70 → REPORT_ONLY: Sinyal var ama güçlü değil
        - ≥ 70  → FULL_TRADE: Güçlü sinyal, emir gönderilebilir
        """
        if ic_score < self.gate_cfg.no_trade:
            return GateAction.NO_TRADE
        elif ic_score < self.gate_cfg.full_trade:
            return GateAction.REPORT_ONLY
        else:
            return GateAction.FULL_TRADE
    
    # =========================================================================
    # PROMPT BUILDER
    # =========================================================================
    
    def _build_prompt(self, data: AIAnalysisInput) -> str:
        """
        Gemini için structured prompt oluşturur.
        
        Prompt tasarım prensipleri:
        - Türkçe yanıt (kullanıcı beklentisi)
        - JSON formatında çıktı (parse edilebilir)
        - Rol tanımı: Kantitatif analist
        - Bağlam: IC analiz sonuçları + risk + market
        - Kısıtlamalar: Sadece LONG/SHORT/WAIT
        """
        # Kategori indikatörleri formatla
        cat_text = ""
        for cat in ['trend', 'momentum', 'volatility', 'volume']:
            if cat in data.category_tops:
                ind = data.category_tops[cat]
                cat_text += f"  - {cat.title()}: {ind['name']} (IC={ind['ic']:+.3f})\n"
            else:
                cat_text += f"  - {cat.title()}: Anlamlı sinyal yok\n"
        
        # TF sıralaması formatla
        tf_text = ""
        for r in data.tf_rankings[:4]:
            tf_text += f"  - {r['tf']}: Skor={r['score']:.0f}, Yön={r['direction']}\n"
        
        prompt = f"""Sen bir kantitatif kripto analisti ve risk yöneticisisin.
Aşağıdaki IC (Information Coefficient) analiz sonuçlarını değerlendir ve işlem kararı ver.

## MARKET VERİSİ
- Coin: {data.coin}
- Fiyat: ${data.price:,.2f}
- 24h Değişim: {data.change_24h:+.1f}%
- 24h Hacim: ${data.volume_24h:,.0f}
- Market Rejimi: {data.market_regime}
- 24h Volatilite: %{data.volatility:.1f}

## IC ANALİZ SONUÇLARI
- En İyi Timeframe: {data.best_timeframe}
- IC Güven Skoru: {data.ic_confidence:.0f}/100
- IC Yönü: {data.ic_direction}
- Kategori Sinyalleri:
{cat_text}
- Timeframe Sıralaması:
{tf_text}

## RİSK HESAPLAMALARI
- ATR: ${data.atr:,.4f} (%{data.atr_pct:.2f})
- Önerilen SL: ${data.sl_price:,.2f}
- Önerilen TP: ${data.tp_price:,.2f}
- Risk/Reward: {data.risk_reward:.1f}
- Pozisyon: {data.position_size:.4f} ({data.leverage}x kaldıraç)

## KURALLAR
1. IC skoru {data.ic_confidence:.0f}/100 — {self.gate_cfg.full_trade} üstünde güçlü sinyal.
2. IC yönü ({data.ic_direction}) ile kararın UYUMLU olmalı. IC SHORT diyorsa LONG verme.
3. Kategori sinyallerinin çoğunluğu aynı yönde olmalı.
4. Ranging/volatile rejimde daha temkinli ol, güveni düşür.
5. ATR çarpanı: Yüksek volatilitede 2.0, düşükte 1.0, normal 1.5 öner.

## ÇIKTI FORMATI
Yanıtını SADECE aşağıdaki JSON formatında ver, başka hiçbir şey yazma:
```json
{{
  "decision": "LONG" | "SHORT" | "WAIT",
  "confidence": 0-100,
  "reasoning": "Türkçe 1-2 cümle gerekçe",
  "atr_multiplier": 1.0-3.0
}}
```"""
        
        return prompt
    
    # =========================================================================
    # GEMİNİ API ÇAĞRISI (RETRY + FALLBACK)
    # =========================================================================
    
    def _call_gemini(self, prompt: str) -> str:
        """
        Gemini API'yi çağırır. Başarısız olursa fallback modelleri dener.
        
        Retry stratejisi:
        1. Ana model (gemini-2.5-flash) → max_retries kez dene
        2. Başarısız → fallback modelleri sırayla dene
        3. Tümü başarısız → exception fırlat
        
        Returns:
        -------
        str
            Gemini'nin ham text yanıtı
        """
        genai = self._get_client()
        
        # Denenecek modeller: ana + fallback'ler
        models_to_try = [self.ai_cfg.model] + self.ai_cfg.fallback_models
        
        last_error = None
        
        for model_name in models_to_try:
            for attempt in range(self.ai_cfg.max_retries):
                try:
                    model = genai.GenerativeModel(
                        model_name=model_name,
                        generation_config={
                            'temperature': self.ai_cfg.temperature,
                            'max_output_tokens': 500,      # JSON yanıt kısa olacak
                        }
                    )
                    
                    response = model.generate_content(prompt)
                    
                    if response and response.text:
                        logger.info(f"Gemini yanıt aldı ({model_name}, deneme {attempt+1})")
                        return response.text
                    
                except Exception as e:
                    last_error = e
                    logger.warning(
                        f"Gemini hatası ({model_name}, deneme {attempt+1}): {e}"
                    )
                    time.sleep(1 * (attempt + 1))          # Exponential backoff
        
        raise ConnectionError(
            f"Gemini API tüm denemeler başarısız. Son hata: {last_error}"
        )
    
    # =========================================================================
    # RESPONSE PARSER
    # =========================================================================
    
    def _parse_response(self, raw: str) -> Dict[str, Any]:
        """
        Gemini'nin JSON yanıtını parse eder.
        
        Gemini bazen markdown code block içinde JSON döndürür:
        ```json
        {"decision": "SHORT", ...}
        ```
        
        Bu fonksiyon hem düz JSON'ı hem code block'u handle eder.
        
        Returns:
        -------
        Dict
            {'decision': 'LONG'|'SHORT'|'WAIT', 'confidence': 0-100,
             'reasoning': str, 'atr_multiplier': float}
        """
        text = raw.strip()
        
        # Markdown code block temizle
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0].strip()
        elif '```' in text:
            text = text.split('```')[1].split('```')[0].strip()
        
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Son çare: text içinde {...} bul
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1:
                try:
                    parsed = json.loads(text[start:end+1])
                except json.JSONDecodeError:
                    logger.error(f"JSON parse başarısız: {text[:200]}")
                    return self._default_response()
            else:
                logger.error(f"JSON bulunamadı: {text[:200]}")
                return self._default_response()
        
        # Doğrulama
        return self._validate_parsed(parsed)
    
    def _validate_parsed(self, parsed: Dict) -> Dict:
        """Parse edilen JSON'ın geçerliliğini kontrol eder."""
        
        # Decision kontrolü
        decision = parsed.get('decision', 'WAIT').upper()
        if decision not in ['LONG', 'SHORT', 'WAIT']:
            decision = 'WAIT'
        
        # Confidence aralık kontrolü
        confidence = float(parsed.get('confidence', 50))
        confidence = max(0, min(100, confidence))
        
        # ATR multiplier aralık kontrolü
        atr_mult = float(parsed.get('atr_multiplier', 1.5))
        atr_mult = max(1.0, min(3.0, atr_mult))
        
        return {
            'decision': decision,
            'confidence': confidence,
            'reasoning': str(parsed.get('reasoning', 'Gerekçe belirtilmedi')),
            'atr_multiplier': atr_mult
        }
    
    def _default_response(self) -> Dict:
        """Parse başarısız olursa varsayılan yanıt."""
        return {
            'decision': 'WAIT',
            'confidence': 30,
            'reasoning': 'AI yanıtı parse edilemedi, güvenli tarafta kal.',
            'atr_multiplier': 1.5
        }
    
    # =========================================================================
    # IC FALLBACK (Gemini başarısız olursa)
    # =========================================================================
    
    def _ic_fallback(self, data: AIAnalysisInput) -> AIDecisionResult:
        """
        Gemini API başarısız olduğunda IC sonuçlarına dayanan fallback karar.
        
        Mantık:
        - IC güven ≥ 70 ve yön net → IC yönünü takip et (düşük güvenle)
        - Diğer durumlar → WAIT
        """
        if data.ic_confidence >= 70 and data.ic_direction in ['LONG', 'SHORT']:
            decision = AIDecision[data.ic_direction]
            confidence = min(data.ic_confidence * 0.7, 65)  # Max %65 güven
            reasoning = (
                f"⚠️ Gemini yanıt veremedi. IC fallback: "
                f"{data.ic_direction} (IC={data.ic_confidence:.0f})"
            )
        else:
            decision = AIDecision.WAIT
            confidence = 20
            reasoning = "Gemini yanıt veremedi ve IC skoru yetersiz."
        
        return AIDecisionResult(
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            gate_action=GateAction.REPORT_ONLY,  # Fallback'te hiç FULL_TRADE verme
            ic_score=data.ic_confidence,
            entry_price=data.price,
            sl_price=data.sl_price,
            tp_price=data.tp_price,
            risk_reward=data.risk_reward,
            model_used="ic_fallback",
        )
    
    # =========================================================================
    # TEK SATIRLIK YARDIMCILAR
    # =========================================================================
    
    def is_available(self) -> bool:
        """Gemini API kullanılabilir mi?"""
        return self.ai_cfg.is_configured()
    
    def get_model_name(self) -> str:
        """Aktif model adı."""
        return self.ai_cfg.model


# =============================================================================
# BAĞIMSIZ ÇALIŞTIRMA TESTİ
# =============================================================================

if __name__ == "__main__":
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    print("=" * 65)
    print("  🤖 AI ENTRY OPTİMİZER — BAĞIMSIZ TEST")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)
    
    optimizer = GeminiOptimizer()
    
    # API durumu
    print(f"\n  Gemini API: {'✅ Yapılandırılmış' if optimizer.is_available() else '❌ Key eksik'}")
    print(f"  Model: {optimizer.get_model_name()}")
    
    # Test input
    test_input = AIAnalysisInput(
        symbol='SOL/USDT:USDT',
        coin='SOL',
        price=185.00,
        change_24h=-2.3,
        best_timeframe='1h',
        ic_confidence=75.0,
        ic_direction='SHORT',
        category_tops={
            'trend': {'name': 'SUPERTREND', 'ic': -0.12},
            'momentum': {'name': 'RSI_14', 'ic': -0.08},
            'volatility': {'name': 'ATR_14', 'ic': 0.05},
            'volume': {'name': 'CMF_20', 'ic': -0.10},
        },
        tf_rankings=[
            {'tf': '1h', 'score': 75, 'direction': 'SHORT'},
            {'tf': '30m', 'score': 68, 'direction': 'SHORT'},
            {'tf': '4h', 'score': 55, 'direction': 'NEUTRAL'},
        ],
        atr=3.70,
        atr_pct=2.0,
        sl_price=188.70,
        tp_price=179.45,
        risk_reward=1.5,
        position_size=0.405,
        leverage=4,
        market_regime='trending',
        volume_24h=500_000_000,
        volatility=4.0
    )
    
    if optimizer.is_available():
        print("\n  📤 Gemini'ye gönderiliyor...")
        decision = optimizer.get_decision(test_input)
        print(f"\n{decision.summary()}")
        print(f"\n  Execute: {'✅ EVET' if decision.should_execute() else '❌ HAYIR'}")
    else:
        print("\n  ⚠️ GEMINI_API_KEY eksik — Gate Keeper testi yapılıyor...")
        
        # Gate keeper testi (API key gerekmez)
        for ic_score in [40, 60, 80]:
            gate = optimizer._check_gate(ic_score)
            print(f"  IC={ic_score} → {gate.value}")
        
        # Fallback testi
        print("\n  IC Fallback testi:")
        test_input.ic_confidence = 75
        fb = optimizer._ic_fallback(test_input)
        print(f"  Karar: {fb.decision.value} | Güven: {fb.confidence:.0f}")
    
    print(f"\n{'=' * 65}")
    print(f"  ✅ TEST TAMAMLANDI")
    print(f"{'=' * 65}")
