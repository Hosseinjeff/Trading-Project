//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   // Define file name only (MetaTrader defaults to MQL5/Files directory)
   string fileName = "data.csv";

   // Attempt to open the file
   int handle = FileOpen(fileName, FILE_WRITE | FILE_CSV);
   if(handle < 0)
   {
      Print("Error opening file: ", GetLastError());
      return INIT_FAILED;
   }

   // Log success
   Print("File opened successfully: ", fileName);

   // Write header to CSV
   FileWrite(handle, "Time", "Open", "High", "Low", "Close");

   // Declare array to hold historical data
   MqlRates rates[];
   int rates_total = CopyRates(Symbol(), PERIOD_M1, 0, 10000, rates);
   if(rates_total > 0)
   {
      // Write data to file
      for(int i = 0; i < rates_total; i++)
      {
         FileWrite(handle, rates[i].time, rates[i].open, rates[i].high, rates[i].low, rates[i].close);
      }
      Print("Data written to file successfully.");
   }
   else
   {
      Print("Error retrieving rates: ", GetLastError());
   }

   // Close file
   FileClose(handle);

   return INIT_SUCCEEDED;
}
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("Expert deinitialized.");
}
//+------------------------------------------------------------------+
