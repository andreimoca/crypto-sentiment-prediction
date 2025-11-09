# Data Directory

This directory should contain the datasets used for the cryptocurrency prediction project.

## Required Data Files

### 1. Market Data: `market_data.csv`

**Format:**
```csv
timestamp,open,high,low,close,volume
2022-01-01 00:00:00,46311.52,46429.18,46249.87,46384.91,1234567.89
2022-01-01 01:00:00,46384.91,46502.34,46301.12,46445.23,1345678.90
...
```

**Requirements:**
- Hourly BTC-USD OHLCV data
- At least 3-6 months of data for meaningful analysis
- Timestamps in UTC
- No missing hours (fill forward if necessary)

**Data Sources:**

1. **Kaggle** (recommended for reproducibility):
   - Search for "Bitcoin Historical Data"
   - Download hourly or minute data
   - Resample to 1-hour if needed

2. **Yahoo Finance** (free, easy):
   ```python
   import yfinance as yf
   btc = yf.download('BTC-USD', start='2022-01-01', end='2022-12-31', interval='1h')
   btc.reset_index().to_csv('market_data.csv', index=False)
   ```

3. **CryptoCompare API** (requires API key):
   - https://www.cryptocompare.com/
   - Free tier available

4. **Binance API**:
   ```python
   from binance.client import Client
   # Historical kline/candlestick data
   ```

### 2. Social Media Data: `reddit_posts.csv`

**Format:**
```csv
timestamp,text
2022-01-01 00:15:23,"Bitcoin looking strong today!"
2022-01-01 00:32:41,"Worried about the correction..."
...
```

**Requirements:**
- Reddit posts from r/Bitcoin and/or r/CryptoCurrency
- Post titles (minimum) or full text (better)
- Timestamps must align with market data period
- At least 10-20 posts per hour on average

**Data Sources:**

1. **Kaggle** (recommended):
   - Search for "cryptocurrency reddit" or "bitcoin reddit posts"
   - Pre-collected datasets available

2. **Pushshift Reddit Archive** (free):
   - https://pushshift.io/
   - Historical Reddit data
   - Example using PSAW library:
   ```python
   from psaw import PushshiftAPI
   api = PushshiftAPI()
   
   # Get submissions from r/Bitcoin
   submissions = api.search_submissions(
       subreddit='Bitcoin',
       after=int(start_date.timestamp()),
       before=int(end_date.timestamp())
   )
   ```

3. **Reddit API (PRAW)** (requires Reddit account):
   ```python
   import praw
   
   reddit = praw.Reddit(
       client_id='YOUR_ID',
       client_secret='YOUR_SECRET',
       user_agent='YOUR_AGENT'
   )
   
   subreddit = reddit.subreddit('Bitcoin')
   posts = []
   for submission in subreddit.new(limit=1000):
       posts.append({
           'timestamp': submission.created_utc,
           'text': submission.title
       })
   ```

4. **PulseReddit Dataset** (research):
   - Referenced in the paper
   - High-frequency cryptocurrency trading signals
   - Check arXiv: 2506.03861

## Data Collection Scripts

### Download BTC Data from Yahoo Finance

Create a file `collect_market_data.py`:

```python
import yfinance as yf
import pandas as pd

# Download 1 year of hourly data
btc = yf.download('BTC-USD', 
                  start='2022-01-01', 
                  end='2022-12-31', 
                  interval='1h')

# Reset index and rename columns
btc = btc.reset_index()
btc.columns = ['timestamp', 'open', 'high', 'low', 'close', 'adj_close', 'volume']

# Drop adj_close (not needed)
btc = btc.drop('adj_close', axis=1)

# Save
btc.to_csv('data/market_data.csv', index=False)
print(f"Downloaded {len(btc)} hourly bars")
```

### Collect Reddit Data

Create a file `collect_reddit_data.py`:

```python
import praw
import pandas as pd
from datetime import datetime

# Configure Reddit API
reddit = praw.Reddit(
    client_id='YOUR_CLIENT_ID',
    client_secret='YOUR_CLIENT_SECRET',
    user_agent='crypto_research'
)

posts = []

# Collect from multiple subreddits
for subreddit_name in ['Bitcoin', 'CryptoCurrency']:
    subreddit = reddit.subreddit(subreddit_name)
    
    # Get recent posts
    for submission in subreddit.new(limit=1000):
        posts.append({
            'timestamp': datetime.fromtimestamp(submission.created_utc),
            'text': submission.title,
            'subreddit': subreddit_name,
            'score': submission.score
        })

# Convert to DataFrame
df = pd.DataFrame(posts)
df = df.sort_values('timestamp')

# Save
df.to_csv('data/reddit_posts.csv', index=False)
print(f"Collected {len(df)} posts")
```

## Alternative: Use Synthetic Data for Testing

If you just want to test the pipeline, run the case study with synthetic data:

```bash
python run_case_study.py
```

This will generate realistic synthetic data automatically.

## Data Privacy and Ethics

- **Reddit data**: Publicly available, but respect Reddit's terms of service
- **No personal information**: Use only post titles/text, not usernames
- **Academic use**: Clearly state this is for research purposes
- **Reproducibility**: Document exact data sources and collection dates

## Recommended Periods for Case Study

Good periods for Bitcoin analysis:

1. **January-March 2022**: 
   - Crash from $47k to $35k
   - Good volatility for testing

2. **May-July 2022**:
   - Terra/LUNA collapse
   - High sentiment activity

3. **November 2022**:
   - FTX collapse
   - Extreme market reactions

4. **2024 periods**:
   - Recent data for most relevance
   - Bitcoin halving effects

## File Size Expectations

- `market_data.csv`: ~5-10 MB for 1 year hourly data
- `reddit_posts.csv`: ~20-50 MB for 1 year with 20 posts/hour

## Troubleshooting

**Q: Where do I get API keys?**
- Yahoo Finance: No key needed
- Reddit: https://www.reddit.com/prefs/apps (create app)
- CryptoCompare: https://www.cryptocompare.com/

**Q: Data has gaps (missing hours)**
- Use forward fill: `df.ffill()`
- Or linear interpolation: `df.interpolate()`

**Q: Too much data to download**
- Start with 3 months for case study
- Expand to 12 months for full experiments

**Q: Reddit API rate limits**
- PRAW: ~60 requests/minute
- Use Pushshift for historical data (no limits)
- Cache results locally

## Next Steps

Once you have the data files:

1. Place `market_data.csv` and `reddit_posts.csv` in this directory
2. Run feature engineering: `python -c "from features import FeatureEngineering; ..."`
3. Run complete case study: `python run_case_study.py`
4. Check generated visualizations and results

