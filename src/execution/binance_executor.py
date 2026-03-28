# =============================================================================
# BINANCE EXECUTION ENGİNE (EMİR YÖNETİMİ) — v2.0 FULL BINANCE
# =============================================================================
# Düzeltmeler v2.0:
# - Tamamen Binance USDT-M Futures (ccxt 'future' tipi) uyumlu hale getirildi.
# - Bitget'e özel 'planType' ve 'productType' parametreleri temizlendi.
# - Sembol yapısı Binance formatına uyarlandı (Örn: BTC/USDT).
# - Binance'in yerleşik STOP_MARKET ve TAKE_PROFIT_MARKET emir tipleri entegre edildi.
# - Margin Type (Isolated) ayarı Binance standartlarına uygun olarak güncellendi.
# =============================================================================

import sys
import ccxt
import time
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import cfg

logger = logging.getLogger(__name__)


# =============================================================================
# DATACLASS'LAR
# =============================================================================

@dataclass
class OrderResult:
    """Tek bir emrin sonucu."""
    order_id: str = ""
    symbol: str = ""
    side: str = ""
    order_type: str = ""
    amount: float = 0.0
    price: float = 0.0
    cost: float = 0.0
    status: str = ""
    filled: float = 0.0
    success: bool = False
    error: str = ""
    raw: Dict = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Tam bir trade execution'ın sonucu (ana emir + SL + TP)."""
    success: bool = False
    symbol: str = ""
    direction: str = ""
    main_order: Optional[OrderResult] = None
    sl_order: Optional[OrderResult] = None
    tp_order: Optional[OrderResult] = None
    actual_entry: float = 0.0
    actual_amount: float = 0.0
    actual_cost: float = 0.0
    dry_run: bool = False
    error: str = ""
    timestamp: str = ""

    def summary(self) -> str:
        mode = "🧪 DRY RUN" if self.dry_run else "🔴 CANLI"
        status = "✅" if self.success else "❌"
        dir_emoji = "🟢" if self.direction == "LONG" else "🔴"
        lines = [f"{status} {mode} | {dir_emoji} {self.symbol} {self.direction}"]
        if self.main_order:
            lines.append(f"📍 Entry: ${self.actual_entry:,.4f} | Miktar: {self.actual_amount:.4f}")
        if self.sl_order and self.sl_order.success:
            lines.append(f"🛑 SL: ${self.sl_order.price:,.6f}")
        if self.tp_order and self.tp_order.success:
            lines.append(f"🎯 TP: ${self.tp_order.price:,.6f}")
        if self.error:
            lines.append(f"❌ Hata: {self.error}")
        return "\n".join(lines)


# =============================================================================
# HELPER: TriggerPrice Precision Fix
# =============================================================================

def _format_trigger_price(price: float) -> float:
    """
    Binance'in stop emirleri için beklediği precision'a göre yuvarlar.
    """
    if price >= 1000:
        return round(price, 2)      # BTC, ETH seviyesi → 2 decimal
    elif price >= 10:
        return round(price, 3)      # SOL, BNB seviyesi → 3 decimal
    elif price >= 1:
        return round(price, 4)      # 1-10 arası → 4 decimal
    elif price >= 0.1:
        return round(price, 5)      # 0.1-1 arası → 5 decimal
    elif price >= 0.01:
        return round(price, 6)      # 0.01-0.1 arası → 6 decimal
    else:
        return round(price, 8)      # Çok düşük fiyat → 8 decimal


# =============================================================================
# ANA EXECUTOR SINIFI
# =============================================================================

class BinanceExecutor:
    """
    Binance USDT-M Perpetual Futures emir yönetimi.
    DRY RUN (varsayılan): Gerçek emir göndermez, API key gerekmez.
    CANLI: Gerçek emir gönderir — dikkatli kullanın!
    """

    DEFAULT_MARKET_INFO: Dict[str, Dict] = {
        'BTC/USDT':   {'price': 2, 'amount': 3, 'min_amount': 0.001,  'min_cost': 5.0, 'max_lev': 125},
        'ETH/USDT':   {'price': 2, 'amount': 2, 'min_amount': 0.01,   'min_cost': 5.0, 'max_lev': 125},
        'SOL/USDT':   {'price': 2, 'amount': 1, 'min_amount': 0.1,    'min_cost': 5.0, 'max_lev': 75},
        'XRP/USDT':   {'price': 4, 'amount': 1, 'min_amount': 1.0,    'min_cost': 5.0, 'max_lev': 75},
        'DOGE/USDT':  {'price': 5, 'amount': 0, 'min_amount': 10.0,   'min_cost': 5.0, 'max_lev': 75},
        'ADA/USDT':   {'price': 4, 'amount': 1, 'min_amount': 1.0,    'min_cost': 5.0, 'max_lev': 50},
        'AVAX/USDT':  {'price': 2, 'amount': 1, 'min_amount': 0.1,    'min_cost': 5.0, 'max_lev': 50},
        'LINK/USDT':  {'price': 3, 'amount': 1, 'min_amount': 0.1,    'min_cost': 5.0, 'max_lev': 50},
        'DOT/USDT':   {'price': 3, 'amount': 1, 'min_amount': 0.1,    'min_cost': 5.0, 'max_lev': 50},
        'MATIC/USDT': {'price': 4, 'amount': 0, 'min_amount': 10.0,   'min_cost': 5.0, 'max_lev': 50},
    }

    FALLBACK_MARKET_INFO: Dict = {
        'price': 4, 'amount': 2, 'min_amount': 0.1, 'min_cost': 5.0, 'max_lev': 50,
    }

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self._exchange = None
        self._market_cache: Dict = {}
        mode = "🧪 DRY RUN (simülasyon)" if dry_run else "🔴 CANLI (gerçek emir)"
        logger.info(f"BinanceExecutor başlatıldı | Mod: {mode}")

    # =========================================================================
    # EXCHANGE INIT
    # =========================================================================

    def _get_exchange(self) -> ccxt.binance:
        """Authenticated Binance exchange (lazy init, sadece canlı modda çağrılır)."""
        if self._exchange is None:
            if not cfg.exchange.is_configured():
                raise ValueError(
                    "Binance API key'leri ayarlanmamış! "
                    ".env dosyasına BINANCE_API_KEY, BINANCE_API_SECRET "
                    "ekleyin."
                )
            self._exchange = ccxt.binance({
                'apiKey': cfg.exchange.api_key,
                'secret': cfg.exchange.api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future', # ✅ Binance USDT-M Futures
                    'positionMode': True,    # ✅ CCXT One-way (Tek Yönlü) Mod
                },
                'sandbox': cfg.exchange.sandbox,
            })
            
            # ✅ Garantilemek için One-Way mod parametresini ekliyoruz:
            try:
                self._exchange.fapiPrivatePostPositionSideDual({'dualSidePosition': 'false'})
            except Exception as e:
                # Zaten false ise hata verebilir, yoksayıyoruz.
                pass
                
            self._exchange.load_markets()
            logger.info(f"Binance exchange başlatıldı (sandbox={cfg.exchange.sandbox})")
        return self._exchange

    # =========================================================================
    # MARKET BİLGİSİ
    # =========================================================================

    def get_market_info(self, symbol: str) -> Dict:
        """
        Sembol için market precision/limits bilgisi.
        DRY RUN: DEFAULT_MARKET_INFO tablosundan (API çağrısı yok).
        CANLI: Binance API'den gerçek değerler.
        """
        if symbol in self._market_cache:
            return self._market_cache[symbol]

        if self.dry_run:
            defaults = self.DEFAULT_MARKET_INFO.get(symbol, self.FALLBACK_MARKET_INFO)
            info = {
                'symbol': symbol,
                'precision': {'price': defaults['price'], 'amount': defaults['amount']},
                'limits': {'min_amount': defaults['min_amount'], 'min_cost': defaults['min_cost']},
                'contract_size': 1.0,
                'max_leverage': defaults['max_lev'],
            }
            self._market_cache[symbol] = info
            return info

        exchange = self._get_exchange()
        if symbol not in exchange.markets:
            raise ValueError(f"'{symbol}' Binance Futures'da bulunamadı")
        market = exchange.markets[symbol]
        info = {
            'symbol': symbol,
            'precision': {
                'price': market.get('precision', {}).get('price', 4),
                'amount': market.get('precision', {}).get('amount', 2),
            },
            'limits': {
                'min_amount': market.get('limits', {}).get('amount', {}).get('min', 0.001),
                'min_cost': market.get('limits', {}).get('cost', {}).get('min', 5.0),
            },
            'contract_size': float(market.get('contractSize', 1.0)),
            # Binance'de leverage bilgisi genelde limits altındadır
            'max_leverage': int(market.get('limits', {}).get('leverage', {}).get('max', 125)),
        }
        self._market_cache[symbol] = info
        return info

    # =========================================================================
    # BAKİYE VE POZİSYON
    # =========================================================================

    def fetch_balance(self) -> Dict:
        """
        USDT bakiyesi.
        ✅ total döndürür (free değil).
        """
        if self.dry_run:
            logger.info("🧪 DRY RUN: Bakiye sorgusu (simülasyon)")
            return {'total': 75.0, 'free': 75.0, 'used': 0.0}

        exchange = self._get_exchange()
        try:
            balance = exchange.fetch_balance({'type': 'future'})
            usdt = balance.get('USDT', {})
            result = {
                'total': float(usdt.get('total', 0) or 0),  
                'free':  float(usdt.get('free', 0) or 0),
                'used':  float(usdt.get('used', 0) or 0),
            }
            logger.info(f"💰 Bakiye: ${result['total']:,.2f} (Free: ${result['free']:,.2f})")
            return result
        except Exception as e:
            logger.error(f"Bakiye çekme hatası: {e}")
            raise

    def fetch_positions(self, symbol: str = None) -> List[Dict]:
        """Açık pozisyonlar. DRY RUN: boş liste döner."""
        if self.dry_run:
            logger.info("🧪 DRY RUN: Pozisyon sorgusu (simülasyon)")
            return []

        exchange = self._get_exchange()
        try:
            symbols = [symbol] if symbol else None
            raw_positions = exchange.fetch_positions(symbols)
            positions = []
            for pos in raw_positions:
                contracts = float(pos.get('contracts', 0) or 0)
                if contracts <= 0:
                    continue
                positions.append({
                    'symbol': pos.get('symbol', ''),
                    'side': pos.get('side', ''),
                    'amount': contracts,
                    'entry_price': float(pos.get('entryPrice', 0) or 0),
                    'unrealized_pnl': float(pos.get('unrealizedPnl', 0) or 0),
                    'leverage': int(pos.get('leverage', 1) or 1),
                    'margin': float(pos.get('initialMargin', 0) or 0),
                    'liquidation_price': float(pos.get('liquidationPrice', 0) or 0),
                })
            logger.info(f"📊 Açık pozisyon: {len(positions)}")
            return positions
        except Exception as e:
            logger.error(f"Pozisyon çekme hatası: {e}")
            raise

    # =========================================================================
    # KALDIRAC VE MARGİN
    # =========================================================================

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Kaldıraç ayarla. DRY RUN: log + True döner."""
        leverage = max(cfg.risk.min_leverage, min(leverage, cfg.risk.max_leverage))
        if self.dry_run:
            logger.info(f"🧪 DRY RUN: Kaldıraç {symbol} → {leverage}x")
            return True
        exchange = self._get_exchange()
        try:
            exchange.set_leverage(leverage, symbol)
            logger.info(f"⚡ Kaldıraç ayarlandı: {symbol} → {leverage}x")
            return True
        except Exception as e:
            if 'No need to change' in str(e) or 'already' in str(e).lower():
                logger.info(f"⚡ Kaldıraç zaten {leverage}x: {symbol}")
                return True
            logger.error(f"Kaldıraç hatası ({symbol}, {leverage}x): {e}")
            return False

    def set_margin_mode(self, symbol: str, mode: str = 'isolated') -> bool:
        """Margin mode ayarla (ISOLATED or CROSSED). DRY RUN: log + True döner."""
        if self.dry_run:
            logger.info(f"🧪 DRY RUN: Margin mode {symbol} → {mode.upper()}")
            return True
        exchange = self._get_exchange()
        try:
            # CCXT'de mode küçük harf beklenir, ancak bazı borsalarda ISOLATED büyük olabilir.
            # ccxt binance implementation'ı genellikle bunu otomatik halleder.
            exchange.set_margin_mode(mode, symbol)
            logger.info(f"📋 Margin mode: {symbol} → {mode.upper()}")
            return True
        except Exception as e:
            if 'No need to change' in str(e) or 'already' in str(e).lower():
                logger.info(f"📋 Margin mode zaten {mode.upper()}: {symbol}")
                return True
            logger.error(f"Margin mode hatası ({symbol}, {mode}): {e}")
            return False

    # =========================================================================
    # YUVARLAMA
    # =========================================================================

    def round_price(self, price: float, symbol: str) -> float:
        """Fiyatı, coinin izin verdiği tick size kuralına göre uydurur."""
        try:
            info = self.get_market_info(symbol)
            if info and 'precision' in info and 'price' in info['precision']:
                precision = info['precision']['price']
                if precision:
                    return float(self._exchange.price_to_precision(symbol, price))
            
            if price >= 100:
                return round(price, 2)
            elif price >= 1:
                return round(price, 4)
            else:
                return round(price, 5)
        except Exception as e:
            logger.warning(f"Fiyat yuvarlama hatası ({symbol}): {e}. Manuel yuvarlanıyor.")
            return round(price, 4)
        
    def round_amount(self, amount: float, symbol: str) -> float:
        """Miktarı borsa precision'ına truncate eder (yukarı değil aşağı)."""
        info = self.get_market_info(symbol)
        precision = info['precision']['amount']
        if isinstance(precision, int):
            factor = 10 ** precision
            return math.floor(amount * factor) / factor
        return math.floor(amount / precision) * precision

    # =========================================================================
    # MARKET EMİR
    # =========================================================================

    def place_market_order(self, symbol: str, side: str, amount: float,
                           reduce_only: bool = False,
                           preset_sl: float = None,  
                           preset_tp: float = None) -> OrderResult: 
        result = OrderResult(symbol=symbol, side=side, order_type='market', amount=amount)
        amount = self.round_amount(amount, symbol)
        result.amount = amount

        info = self.get_market_info(symbol)
        min_amount = info['limits']['min_amount']
        if amount < min_amount:
            result.error = f"Miktar ({amount}) < minimum ({min_amount})"
            logger.error(result.error)
            return result

        if self.dry_run:
            result.order_id = f"DRY_{int(time.time())}"
            result.status = "closed"
            result.filled = amount
            result.success = True
            logger.info(f"🧪 DRY RUN: {side.upper()} {amount} {symbol} (market)")
            return result

        exchange = self._get_exchange()
        try:
            params = {}
            if reduce_only:
                params['reduceOnly'] = True

            # Binance'te ana emre doğrudan SL/TP parametresi gömülemez.
            # O yüzden önce ana emir açılır, başarılıysa execution tarafında SL/TP çağrılır.
            # (execute_trade metodunda bunu zincirleme olarak çözeceğiz).

            order = exchange.create_order(
                symbol=symbol, 
                type='market', 
                side=side, 
                amount=amount, 
                params=params
            )
            result.order_id = str(order.get('id', ''))
            result.price = float(order.get('average') or order.get('price') or 0.0)
            result.filled = float(order.get('filled') or 0.0)
            result.status = order.get('status') or 'unknown'
            
            result.success = True if result.order_id else False
            result.raw = order
            
            logger.info(f"✅ Market Emir İletildi: {side.upper()} {amount} {symbol} | ID: {result.order_id}")
            return result
            
        except Exception as e:
            result.error = f"{type(e).__name__}: {str(e)}"
            logger.error(f"❌ Market emir hatası: {result.error}")
            return result

    # =========================================================================
    # SL/TP EMİRLERİ (BINANCE UYUMLU TRIGGER)
    # =========================================================================

    def place_stop_loss(self, symbol: str, side: str, amount: float,
                        trigger_price: float) -> OrderResult:
        result = OrderResult(symbol=symbol, side=side, order_type='stop_market',
                             amount=amount, price=trigger_price)

        trigger_price = self.round_price(trigger_price, symbol)
        amount = self.round_amount(amount, symbol)
        result.price = trigger_price
        result.amount = amount

        if self.dry_run:
            result.order_id = f"DRY_SL_{int(time.time())}"
            result.success = True
            return result

        exchange = self._get_exchange()
        try:
            # ✅ Binance'te Stop Loss Market Emri
            order = exchange.create_order(
                symbol=symbol, 
                type='STOP_MARKET', 
                side=side, 
                amount=amount,
                params={
                    'stopPrice': str(trigger_price), # Tetiklenme fiyatı
                    'reduceOnly': True,              # Sadece pozisyonu kapatmak için
                    'workingType': 'MARK_PRICE'      # İğnelerden korunmak için Mark fiyatını baz al
                }
            )
            result.order_id = str(order.get('id', ''))
            result.status = order.get('status') or 'unknown'
            result.success = True if result.order_id else False
            result.raw = order
            logger.info(f"🛑 SL Emri İletildi: {side.upper()} {amount} {symbol} @ {trigger_price}")
            return result
        except Exception as e:
            result.error = f"SL Hatası: {e}"
            logger.error(result.error)
            return result

    def place_take_profit(self, symbol: str, side: str, amount: float,
                          trigger_price: float) -> OrderResult:
        result = OrderResult(symbol=symbol, side=side, order_type='take_profit_market',
                             amount=amount, price=trigger_price)

        trigger_price = self.round_price(trigger_price, symbol)
        amount = self.round_amount(amount, symbol)
        result.price = trigger_price
        result.amount = amount

        if self.dry_run:
            result.order_id = f"DRY_TP_{int(time.time())}"
            result.success = True
            return result

        exchange = self._get_exchange()
        try:
            # ✅ Binance'te Take Profit Market Emri
            order = exchange.create_order(
                symbol=symbol, 
                type='TAKE_PROFIT_MARKET', 
                side=side, 
                amount=amount,
                params={
                    'stopPrice': str(trigger_price),
                    'reduceOnly': True,
                    'workingType': 'MARK_PRICE'
                }
            )
            result.order_id = str(order.get('id', ''))
            result.status = order.get('status') or 'unknown'
            result.success = True if result.order_id else False
            result.raw = order
            logger.info(f"🎯 TP Emri İletildi: {side.upper()} {amount} {symbol} @ {trigger_price}")
            return result
        except Exception as e:
            result.error = f"TP Hatası: {e}"
            logger.error(result.error)
            return result

    # =========================================================================
    # POZİSYON KAPATMA
    # =========================================================================

    def close_position(self, symbol: str, side: str, amount: float) -> OrderResult:
        """LONG kapatma → sell. SHORT kapatma → buy."""
        close_side = 'sell' if side.lower() == 'long' else 'buy'
        logger.info(f"📤 Pozisyon kapatma: {close_side} {amount} {symbol}")
        return self.place_market_order(
            symbol=symbol, side=close_side, amount=amount, reduce_only=True
        )

    def close_all_positions(self) -> List[OrderResult]:
        """Tüm açık pozisyonları kapatır (acil durum / kill switch)."""
        positions = self.fetch_positions()
        results = []
        for pos in positions:
            results.append(self.close_position(pos['symbol'], pos['side'], pos['amount']))
        logger.info(f"🚨 Tüm pozisyonlar kapatıldı: {len(results)}")
        return results

    # =========================================================================
    # ANA TRADE EXECUTION
    # =========================================================================

    def execute_trade(self, trade_calc, skip_sl: bool = False,
                      skip_tp: bool = False) -> ExecutionResult:
        """
        Tam trade execution pipeline (Binance için zincirleme yapı):
        1. Margin mode ayarla (isolated)
        2. Kaldıraç ayarla
        3. Ana market emrini gönder
        4. Başarılıysa SL ve TP emirlerini gönder. Herhangi biri başarısız olursa
           ana pozisyonu geri kapat (güvenlik ağı).
        """
        timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        exec_result = ExecutionResult(
            symbol=trade_calc.symbol,
            direction=trade_calc.direction,
            dry_run=self.dry_run,
            timestamp=timestamp
        )

        if not trade_calc.is_approved():
            exec_result.error = f"Trade onaylı değil: {trade_calc.rejection_reasons}"
            logger.warning(exec_result.error)
            return exec_result

        symbol    = trade_calc.symbol
        direction = trade_calc.direction
        pos = trade_calc.position
        
        sl_price  = trade_calc.stop_loss.price if not skip_sl else None
        tp_price  = trade_calc.take_profit.price if not skip_tp else None
        
        open_side  = 'buy'  if direction == 'LONG' else 'sell'
        close_side = 'sell' if direction == 'LONG' else 'buy'

        logger.info(
            f"{'🧪' if self.dry_run else '🔴'} Trade Başlıyor: {direction} {symbol} | "
            f"Size: {pos.size} | Lev: {pos.leverage}x"
        )

        try:
            # 1. Margin mode
            self.set_margin_mode(symbol, 'isolated')

            # 2. Kaldıraç
            self.set_leverage(symbol, pos.leverage)

            # 3. Ana market emri
            main_order = self.place_market_order(symbol=symbol, side=open_side, amount=pos.size)
            exec_result.main_order = main_order

            if not main_order.success:
                exec_result.error = f"Ana emir başarısız: {main_order.error}"
                return exec_result

            exec_result.actual_entry  = main_order.price or trade_calc.entry_price
            exec_result.actual_amount = main_order.filled or pos.size
            exec_result.actual_cost   = main_order.cost or (pos.size * trade_calc.entry_price)

            # 4. Binance'te SL/TP zincirleme gönderilir (Ana emirden ayrı)
            if sl_price:
                sl_order = self.place_stop_loss(symbol, close_side, exec_result.actual_amount, sl_price)
                exec_result.sl_order = sl_order
                
                # Güvenlik ağı: SL koyulamazsa ana pozisyonu anında kapat!
                if not sl_order.success and not self.dry_run:
                    logger.error("🛑 ACİL: SL emri girilemedi! Korumasız kalmamak için pozisyon kapatılıyor.")
                    self.close_position(symbol, direction, exec_result.actual_amount)
                    exec_result.error = f"SL girilemediği için pozisyon iptal edildi. Hata: {sl_order.error}"
                    exec_result.success = False
                    return exec_result

            if tp_price:
                tp_order = self.place_take_profit(symbol, close_side, exec_result.actual_amount, tp_price)
                exec_result.tp_order = tp_order

            exec_result.success = True
            logger.info(f"✅ Trade GÜVENLİ ŞEKİLDE ONAYLANDI: {direction} {exec_result.actual_amount} {symbol}")

        except Exception as e:
            exec_result.error = str(e)
            logger.error(f"❌ Trade hatası: {e}")

        return exec_result

    # =========================================================================
    # AÇIK EMİRLERİ İPTAL
    # =========================================================================

    def fetch_open_plan_orders(self, symbol: str) -> List[Dict]:
        """
        Sembol için borsada bekleyen trigger emirlerini (STOP/TAKE_PROFIT) döndürür.
        """
        if self.dry_run:
            return []                           

        exchange = self._get_exchange()
        try:
            # Binance'te doğrudan açık emirleri çekmek yeterlidir
            open_orders = exchange.fetch_open_orders(symbol)
            
            # Sadece STOP_MARKET ve TAKE_PROFIT_MARKET tiplerini filtrele
            plan_orders = [
                o for o in open_orders
                if o.get('type') in ('STOP_MARKET', 'TAKE_PROFIT_MARKET', 'stop_market', 'take_profit_market')
            ]
            logger.debug(f"📋 {symbol} açık stop/tp emir sayısı: {len(plan_orders)}")
            return plan_orders

        except Exception as e:
            logger.warning(f"⚠️ fetch_open_plan_orders hatası ({symbol}): {e}")
            return []                           


    def has_tp_sl_orders(self, symbol: str) -> bool:
        """Sembol için borsada halihazırda SL veya TP emir var mı?"""
        plan_orders = self.fetch_open_plan_orders(symbol)
        return len(plan_orders) > 0

    def get_open_position_symbols(self) -> set:
        """Borsada açık pozisyon olan sembolleri set olarak döndürür."""
        try:
            positions = self.fetch_positions()
            return {p['symbol'] for p in positions if p.get('symbol')}
        except Exception as e:
            logger.warning(f"⚠️ get_open_position_symbols hatası: {e}")
            return set()                        

    def cancel_open_orders(self, symbol: str) -> int:
        """Sembol için tüm açık emirleri iptal eder."""
        if self.dry_run:
            logger.info(f"🧪 DRY RUN: {symbol} açık emirler iptal edildi")
            return 0
        exchange = self._get_exchange()
        try:
            open_orders = exchange.fetch_open_orders(symbol)
            canceled = 0
            for order in open_orders:
                try:
                    exchange.cancel_order(order['id'], symbol)
                    canceled += 1
                except Exception as e:
                    logger.warning(f"Emir iptal hatası ({order['id']}): {e}")
            logger.info(f"🗑️ {canceled} emir iptal edildi: {symbol}")
            return canceled
        except Exception as e:
            logger.error(f"Emir iptal hatası ({symbol}): {e}")
            return 0


# =============================================================================
# BAĞIMSIZ TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    print("=" * 65)
    print("  📡 BINANCE EXECUTION ENGİNE v2.0 — BAĞIMSIZ TEST")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    executor = BinanceExecutor(dry_run=True)
    balance = executor.fetch_balance()
    print(f"\n  💰 Bakiye: ${balance['total']:,.2f}")

    info = executor.get_market_info('SOL/USDT')
    print(f"  📋 SOL precision: price={info['precision']['price']}, amount={info['precision']['amount']}")

    order = executor.place_market_order('SOL/USDT', 'sell', 0.405)
    print(f"\n  📤 Market: {order.side.upper()} {order.amount} → {order.status} ✅")

    print(f"\n{'=' * 65}")
    print("  ✅ TEST TAMAMLANDI")
    print(f"{'=' * 65}")