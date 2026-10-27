//+------------------------------------------------------------------+
//| BTC_Martingale_Grid.mq5                                          |
//| Grid martingale for BTCUSD (cent or standard account).           |
//| Same logic as martingale/backtest.py:                            |
//|  - first trade in the direction of an EMA trend filter           |
//|  - add a layer every StepPct% against the basket, lot x LotMult  |
//|  - close the basket at average entry +/- TpPct%                   |
//|  - optional equity stop: close basket at -EquitySLPct% of equity |
//+------------------------------------------------------------------+
#property copyright "Kronos project"
#property version   "1.00"

#include <Trade/Trade.mqh>

enum ENUM_DIR_MODE { DIR_TREND = 0, DIR_LONG_ONLY = 1, DIR_SHORT_ONLY = 2 };

input group "Sizing"
input double InpStartLot     = 0.01;   // First layer lot
input double InpAutoLotPer   = 0;      // >0: 0.01 lot per this much balance (compounding)
input double InpLotMult      = 1.5;    // Martingale lot multiplier
input int    InpMaxLayers    = 14;     // Max layers in a basket

input group "Grid and exit"
input double InpStepPct      = 1.5;    // First grid gap, % of price
input double InpStepMult     = 1.0;    // Each next gap = previous gap x this
input double InpTpPct        = 1.0;    // Basket take profit, % from average entry
input double InpEquitySLPct  = 10.0;   // Equity stop, % of equity at basket start (0 = off)

input group "Direction"
input ENUM_DIR_MODE   InpDirMode = DIR_TREND;
input ENUM_TIMEFRAMES InpEmaTF   = PERIOD_H1;
input int             InpEmaLen  = 200;

input group "Misc"
input int    InpCooldownMin  = 0;      // Minutes to wait after a basket closes
input long   InpMagic        = 20261027;
input int    InpSlippage     = 50;     // points

CTrade   trade;
int      emaHandle = INVALID_HANDLE;
double   basketStartEquity = 0;
double   nextGap = 0;
double   firstLot = 0;
datetime lastClose = 0;

//+------------------------------------------------------------------+
int OnInit()
  {
   trade.SetExpertMagicNumber(InpMagic);
   trade.SetDeviationInPoints(InpSlippage);
   emaHandle = iMA(_Symbol, InpEmaTF, InpEmaLen, 0, MODE_EMA, PRICE_CLOSE);
   if(emaHandle == INVALID_HANDLE)
      return INIT_FAILED;
   RestoreState();
   return INIT_SUCCEEDED;
  }

void OnDeinit(const int reason) { IndicatorRelease(emaHandle); }

//+------------------------------------------------------------------+
//| Basket helpers                                                    |
//+------------------------------------------------------------------+
int CountLayers(int &side, double &volume, double &avgPrice, double &lastPrice, double &lastLot, double &profit)
  {
   int n = 0; side = 0; volume = 0; avgPrice = 0; lastPrice = 0; lastLot = 0; profit = 0;
   datetime newest = 0, oldest = D'3000.01.01';
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol || PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;
      double v = PositionGetDouble(POSITION_VOLUME);
      double p = PositionGetDouble(POSITION_PRICE_OPEN);
      side = PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY ? 1 : -1;
      volume += v; avgPrice += v * p; n++;
      profit += PositionGetDouble(POSITION_PROFIT) + PositionGetDouble(POSITION_SWAP);
      datetime t = (datetime)PositionGetInteger(POSITION_TIME);
      if(t >= newest) { newest = t; lastPrice = p; lastLot = v; }
      if(t <= oldest) { oldest = t; firstLot = v; }
     }
   if(volume > 0) avgPrice /= volume;
   return n;
  }

void CloseBasket()
  {
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol || PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;
      trade.PositionClose(ticket);
     }
   lastClose = TimeCurrent();
  }

double NormalizeLot(double lot)
  {
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double mn   = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double mx   = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   lot = MathRound(lot / step + 1e-9) * step;   // half-up: 0.015 -> 0.02
   return MathMax(mn, MathMin(mx, lot));
  }

double BaseLot()
  {
   if(InpAutoLotPer > 0)
      return NormalizeLot(MathFloor(AccountInfoDouble(ACCOUNT_BALANCE) / InpAutoLotPer) * 0.01);
   return NormalizeLot(InpStartLot);
  }

// After a restart, rebuild the grid gap from the number of open layers
void RestoreState()
  {
   int side; double vol, avg, lastP, lastL, pr;
   int n = CountLayers(side, vol, avg, lastP, lastL, pr);
   basketStartEquity = AccountInfoDouble(ACCOUNT_BALANCE);
   if(n > 0)
      nextGap = lastP * InpStepPct / 100.0 * MathPow(InpStepMult, n - 1);
  }

int TrendSide()
  {
   if(InpDirMode == DIR_LONG_ONLY)  return 1;
   if(InpDirMode == DIR_SHORT_ONLY) return -1;
   double ema[1];
   if(CopyBuffer(emaHandle, 0, 1, 1, ema) != 1) return 0;
   double close1 = iClose(_Symbol, InpEmaTF, 1);
   return close1 >= ema[0] ? 1 : -1;
  }

bool Open(int side, double lot, string comment)
  {
   return side > 0 ? trade.Buy(lot, _Symbol, 0, 0, 0, comment)
                   : trade.Sell(lot, _Symbol, 0, 0, 0, comment);
  }

//+------------------------------------------------------------------+
void OnTick()
  {
   int side; double vol, avg, lastP, lastL, profit;
   int n = CountLayers(side, vol, avg, lastP, lastL, profit);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);

   if(n == 0)
     {
      if(TimeCurrent() - lastClose < InpCooldownMin * 60) return;
      int s = TrendSide();
      if(s == 0) return;
      basketStartEquity = AccountInfoDouble(ACCOUNT_EQUITY);
      double px = s > 0 ? ask : bid;
      firstLot = BaseLot();
      if(Open(s, firstLot, "L1"))
         nextGap = px * InpStepPct / 100.0;
      return;
     }

   // equity stop on the basket
   if(InpEquitySLPct > 0 && -profit >= basketStartEquity * InpEquitySLPct / 100.0)
     {
      PrintFormat("Equity stop: basket loss %.2f with %d layers", profit, n);
      CloseBasket();
      return;
     }

   // basket take profit
   double exitPx = side > 0 ? bid : ask;
   double tpPx   = avg * (1.0 + side * InpTpPct / 100.0);
   if((side > 0 && exitPx >= tpPx) || (side < 0 && exitPx <= tpPx))
     {
      CloseBasket();
      return;
     }

   // add a layer
   if(n < InpMaxLayers)
     {
      double entryPx = side > 0 ? ask : bid;
      if((side > 0 && entryPx <= lastP - nextGap) || (side < 0 && entryPx >= lastP + nextGap))
        {
         // layer k lot = first lot * mult^k (not chained, so rounding never stalls growth)
         double lot = NormalizeLot(firstLot * MathPow(InpLotMult, n));
         if(Open(side, lot, "L" + IntegerToString(n + 1)))
            nextGap *= InpStepMult;
        }
     }
  }
//+------------------------------------------------------------------+
