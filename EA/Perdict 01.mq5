//+------------------------------------------------------------------+
//| Expert Advisor: Predict 01                                       |
//+------------------------------------------------------------------+
#property strict

input string python_path = "C:\\Program Files\\Python310\\python.exe";  // Path to Python executable
input string script_path = "C:\\users\\Setin\\Documents\\GitHub\\Trading-Project\\python\\predict_server.py";  // Path to Python server script
input string server_url = "http://127.0.0.1:5000/predict";  // URL of the prediction server

// Declare variables to manage the Python server
int server_pid = -1;

// Import ShellExecuteW to execute external commands
#import "shell32.dll"
int ShellExecuteW(int hwnd, string operation, string file, string parameters, string directory, int showCmd);
#import

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+

input string flag_file_path = "ea_flag_file.txt";  // Path to the flag file
int OnInit()
  {
   // Create a flag file to indicate that EA is running
   int handle = FileOpen(flag_file_path, FILE_WRITE|FILE_TXT);
   if(handle != INVALID_HANDLE)
     {
      FileWriteString(handle, "EA is running");
      FileClose(handle);
      Print("Flag file created: EA is running.");
     }
   else
     {
      Print("Failed to create flag file.");
      return INIT_FAILED;
     }

   // Start the Python server
   server_pid = StartPythonServer();
   if(server_pid != -1)
     {
      Print("Python server started with PID: ", server_pid);
     }
   else
     {
      Print("Failed to start Python server.");
      return INIT_FAILED;
     }

   return INIT_SUCCEEDED;
  }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   // Delete the flag file to indicate that EA has stopped
   FileDelete(flag_file_path);
   Print("Flag file deleted: EA has stopped.");

   // Stop the Python server
   StopPythonServer();
  }


//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
// Fetch prediction from the Python server
   string prediction = FetchPrediction(server_url);

   if(prediction != "")
     {
      Print("Prediction received: ", prediction);

      // Execute trading logic based on the prediction
      ExecuteTradingLogic(prediction);
     }
   else
     {
      Print("Failed to fetch prediction. Skipping trading logic.");
     }
  }

//+------------------------------------------------------------------+
//| Start the Python server                                          |
//+------------------------------------------------------------------+
int StartPythonServer()
{
    string batch_file = "\"C:\\Users\\Setin\\Documents\\GitHub\\Trading-Project\\run_python.bat\"";
    int result = ShellExecuteW(0, "open", batch_file, "", "", 1);
    
    Print("ShellExecuteW result: ", result);  // Print the result of ShellExecuteW

    if(result > 32)  // Successful execution
    {
        Print("Python server started successfully.");
        return result;  // Return a dummy PID as ShellExecute doesn't return the actual PID
    }
    else
    {
        Print("Failed to execute Python server. Result code: ", result);
        return -1;
    }
}

//+------------------------------------------------------------------+
//| Stop the Python server                                           |
//+------------------------------------------------------------------+
void StopPythonServer()
{
   // Kill any running Python processes (if any exist) before starting the server
   string kill_command = "taskkill /F /IM python.exe";
   int result = ShellExecuteW(0, "open", kill_command, "", "", 0);

   if (result > 32)
   {
       Print("Python server stopped successfully.");
   }
   else
   {
       Print("Failed to stop Python server. Result code: ", result);
   }
}

//+------------------------------------------------------------------+
//| Fetch prediction from the server                                 |
//+------------------------------------------------------------------+
string FetchPrediction(string url)
{
    int timeout = 5000;  // Timeout in milliseconds
    char result[1024];   // This will hold the response data
    string response_body = "";  // Response body as a string
    string response_headers = "";  // Response headers

    // Prepare the data to be sent (e.g., JSON data)
    string jsonData = "{\"features\": [1.2, 3.4, 5.6, 7.8, 9.0, 11.2, 5.6, 7.8, 8.9, 1.0, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0, 1.3, 2.4, 3.5, 4.6, 5.7, 6.8, 7.9, 8.0, 9.1]}"; // Example feature set (replace with actual feature values)

    // Convert jsonData to char array
    char data[];
    StringToCharArray(jsonData, data);

    // Debug: Print request data
    Print("Sending POST request with data: ", jsonData);

    // Send POST request with the JSON data and content-type header
    int res = WebRequest(
        "POST",                          // HTTP method
        url,                             // The server URL
        "Content-Type: application/json\r\n",  // Header for content-type
        "",                              // Additional headers (empty in this case)
        timeout,                         // Timeout in ms
        data,                            // Request body (char[])
        ArraySize(data),                 // Size of the request body
        result,                          // The response (char[])
        response_body                    // Response body (string)
    );
    
    // Check if the request was successful
    if (res != -1)
    {
        // Print the result for debugging
        string resultStr = CharArrayToString(result);  // Convert char[] to string
        Print("Server Response: ", resultStr);  // Log the server response

        // Extract prediction from server response (assuming JSON format)
        string prediction = ParsePredictionFromResponse(resultStr);  // Parse the prediction
        return prediction;
    }
    else
    {
        // WebRequest failed, print the error code
        int error_code = GetLastError();
        Print("WebRequest failed with error code: ", error_code);
        return "";
    }
}


//+------------------------------------------------------------------+
//| Parse the prediction from the response                           |
//+------------------------------------------------------------------+
string ParsePredictionFromResponse(string response)
  {
   string prediction = "";
   int startIdx = StringFind(response, "\"prediction\":");
   if(startIdx != -1)
     {
      startIdx += StringLen("\"prediction\":") + 1;  // Move past "prediction": and space
      int endIdx = StringFind(response, "\"", startIdx);
      if(endIdx != -1)
        {
         prediction = StringSubstr(response, startIdx, endIdx - startIdx);
        }
     }
   else
     {
      Print("Prediction not found in response: ", response);
     }
   return prediction;
  }

//+------------------------------------------------------------------+
//| Execute trading logic based on prediction                        |
//+------------------------------------------------------------------+
void ExecuteTradingLogic(string prediction)
  {
   double lot_size = 0.1;  // Adjust based on risk management rules
   double sl = 10 * _Point;  // Example stop-loss distance
   double tp = 20 * _Point;  // Example take-profit distance

   MqlTradeRequest request;
   MqlTradeResult result;

   ZeroMemory(request);
   ZeroMemory(result);

   request.action = TRADE_ACTION_DEAL;
   request.symbol = _Symbol;
   request.volume = lot_size;
   request.deviation = 5;
   request.type_filling = ORDER_FILLING_IOC;

   if(prediction == "buy")
     {
      request.type = ORDER_TYPE_BUY;
      request.price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      request.sl = request.price - sl;
      request.tp = request.price + tp;
     }
   else
      if(prediction == "sell")
        {
         request.type = ORDER_TYPE_SELL;
         request.price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         request.sl = request.price + sl;
         request.tp = request.price - tp;
        }
      else
        {
         Print("Invalid prediction: ", prediction);
         return;
        }

   if(OrderSend(request, result))
     {
      Print("Order placed successfully: ", result.order);
     }
   else
     {
      Print("Order failed. Error: ", GetLastError());
     }
  }


//+------------------------------------------------------------------+
//| Send trading order                                               |
//+------------------------------------------------------------------+
bool SendOrder(int order_type, double lot_size)
  {
   MqlTradeRequest request;
   MqlTradeResult result;

   ZeroMemory(request);
   ZeroMemory(result);

   request.action = TRADE_ACTION_DEAL;
   request.symbol = _Symbol;
   request.volume = lot_size;
   request.type = ENUM_ORDER_TYPE(order_type); // Explicitly cast to ENUM_ORDER_TYPE
   request.price = (order_type == ORDER_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) : SymbolInfoDouble(_Symbol, SYMBOL_BID);
   request.deviation = 5;
   request.type_filling = ORDER_FILLING_IOC;

   if(OrderSend(request, result))
     {
      Print("Order placed successfully: ", result.order);
      return true;
     }
   else
     {
      Print("Order placement failed. Error code: ", GetLastError());
      return false;
     }
  }
//+------------------------------------------------------------------+
