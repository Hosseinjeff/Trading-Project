TIMEFRAMES = ["M5", "H1", "H4", "D1", "W1"]
INDICATORS = ["EMA_50", "RSI", "MACD", "MACD_signal", "MACD_histogram", "Bollinger_upper", "Bollinger_lower"]

def generate_feature_columns(timeframes, indicators):
    feature_columns = {}
    for tf in timeframes:
        columns = [f"{indicator}_{tf}" for indicator in indicators]
        feature_columns[tf] = columns
    return feature_columns

# Generate the dynamic feature configuration
FEATURE_CONFIG = generate_feature_columns(TIMEFRAMES, INDICATORS)

# Example to print out the generated FEATURE_CONFIG
for timeframe, columns in FEATURE_CONFIG.items():
    print(f"{timeframe}: {columns}")
