"""
[MADDE 12] Coin başına performans hafızası ve cooldown yönetimi.

Her coin için rolling istatistik tutar. Performans bozuk coinleri
geçici olarak universe'den çıkarır (3 gün cooldown).

Mantık:
- Bir coinin son N trade'i WR eşik altıysa → cooldown başlar
- MIN_TRADES_FOR_DECISION yüksek tutulmuştur (10): az örneklemde karar verilmez
- Cooldown sırasında o coin trade edilmez
- 3 gün sonra otomatik olarak universe'e geri döner

Kullanıcı notu: MIN_TRADES_FOR_DECISION=10 — az örneklem = bilinmez,
karar yok. Bu değer arttıkça daha güvenli, azaldıkça daha hızlı reaksiyon.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
from collections import deque

logger = logging.getLogger(__name__)


class CoinPerformanceTracker:
    """
    Coin başına rolling performance ve cooldown yönetimi.

    NOT: MIN_TRADES_FOR_DECISION=10 ile çalışır.
    Az örneklemli coinlerde (< 10 trade) hiç cooldown kararı verilmez.
    Veri birikince otomatik olarak devreye girer.
    """

    # Sınıf seviyesi sabitler
    ROLLING_WINDOW = 30                         # Son kaç trade'e bakılır
    MIN_TRADES_FOR_DECISION = 10               # Karar için minimum trade sayısı
                                                # (kullanıcı isteği: az örneklemde karar verme)
    WR_COOLDOWN_THRESHOLD = 0.25               # WR < %25 → cooldown tetikler
    COOLDOWN_HOURS = 72                        # 3 gün cooldown
    STATE_FILE = "logs/coin_performance.json"  # Disk persistence

    def __init__(self):
        # Her coin için son N trade outcome'u (1=win, 0=loss)
        self.coin_outcomes: Dict[str, deque] = {}
        # Cooldown başlangıç zamanları
        self.cooldown_until: Dict[str, datetime] = {}
        # Diskten yükle
        self._load_state()

    def record_trade(self, symbol: str, pnl: float) -> None:
        """
        Bir trade'in sonucunu kaydet.

        Parameters:
            symbol: Coin sembolü ('BTC/USDT:USDT' veya 'BTC')
            pnl: Net PnL ($) — pozitifse win, değilse loss
        """
        coin = self._normalize_symbol(symbol)
        outcome = 1 if pnl > 0 else 0

        if coin not in self.coin_outcomes:
            self.coin_outcomes[coin] = deque(maxlen=self.ROLLING_WINDOW)

        self.coin_outcomes[coin].append(outcome)

        # Cooldown trigger kontrolü
        self._check_cooldown_trigger(coin)

        # Diske kaydet
        self._save_state()

    def is_coin_allowed(self, symbol: str) -> Tuple[bool, str]:
        """
        Bir coin trade edilebilir mi?

        Returns:
            (allowed, reason): (True, "OK") veya (False, "Cooldown remaining 2.3h")
        """
        coin = self._normalize_symbol(symbol)

        # Cooldown kontrolü
        if coin in self.cooldown_until:
            now = datetime.now(timezone.utc)
            until = self.cooldown_until[coin]

            if now < until:                     # Hâlâ cooldown'da
                remaining = (until - now).total_seconds() / 3600
                return False, f"Cooldown remaining {remaining:.1f}h (WR düşük)"
            else:                               # Cooldown süresi doldu
                del self.cooldown_until[coin]
                logger.info(f"✅ {coin} cooldown sona erdi — universe'e geri döndü")
                self._save_state()

        return True, "OK"

    def _check_cooldown_trigger(self, coin: str) -> None:
        """Coinin son N trade'i kötüyse cooldown başlat."""
        outcomes = self.coin_outcomes[coin]

        if len(outcomes) < self.MIN_TRADES_FOR_DECISION:
            return                              # Yetersiz veri, karar verme

        wr = sum(outcomes) / len(outcomes)

        if wr < self.WR_COOLDOWN_THRESHOLD:
            # Cooldown başlat
            cooldown_end = datetime.now(timezone.utc) + timedelta(hours=self.COOLDOWN_HOURS)
            self.cooldown_until[coin] = cooldown_end

            logger.warning(
                f"🚫 {coin} COOLDOWN — son {len(outcomes)} trade WR={wr:.1%} "
                f"(< {self.WR_COOLDOWN_THRESHOLD:.0%}). "
                f"{self.COOLDOWN_HOURS}h ban."
            )

    def get_stats(self) -> Dict[str, Dict]:
        """Tüm coin'lerin mevcut istatistikleri (rapor için)."""
        stats = {}
        for coin, outcomes in self.coin_outcomes.items():
            wr = sum(outcomes) / len(outcomes) if outcomes else 0
            stats[coin] = {
                'n_trades': len(outcomes),
                'win_rate': wr,
                'cooldown_until': (
                    self.cooldown_until[coin].isoformat()
                    if coin in self.cooldown_until else None
                ),
                'decision_ready': len(outcomes) >= self.MIN_TRADES_FOR_DECISION,
            }
        return stats

    def _normalize_symbol(self, symbol: str) -> str:
        """Sembolü normalize et: 'BTC/USDT:USDT' → 'BTC'"""
        return symbol.split('/')[0]

    def _save_state(self) -> None:
        """State'i diske kaydet."""
        try:
            state_path = Path(self.STATE_FILE)
            state_path.parent.mkdir(parents=True, exist_ok=True)

            state = {
                'outcomes': {k: list(v) for k, v in self.coin_outcomes.items()},
                'cooldown_until': {
                    k: v.isoformat() for k, v in self.cooldown_until.items()
                },
                'saved_at': datetime.now(timezone.utc).isoformat(),
            }

            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.warning(f"⚠️ Coin performance state kayıt hatası: {e}")

    def _load_state(self) -> None:
        """State'i diskten yükle."""
        state_path = Path(self.STATE_FILE)
        if not state_path.exists():
            return

        try:
            with open(state_path, 'r') as f:
                state = json.load(f)

            self.coin_outcomes = {
                k: deque(v, maxlen=self.ROLLING_WINDOW)
                for k, v in state.get('outcomes', {}).items()
            }
            self.cooldown_until = {
                k: datetime.fromisoformat(v)
                for k, v in state.get('cooldown_until', {}).items()
            }
            logger.info(
                f"📂 Coin performance state yüklendi: "
                f"{len(self.coin_outcomes)} coin, "
                f"{len(self.cooldown_until)} cooldown"
            )
        except Exception as e:
            logger.error(f"❌ Coin performance state yükleme hatası: {e}")
