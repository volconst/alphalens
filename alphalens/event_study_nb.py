
# coding: utf-8

# This is an IPython notebook from https://www.quantopian.com/posts/event-study-tearsheet
# Added to git as .py for easier tracking of changes.
# Not runnable as .py, copy in quantopian.com research notebook to run.
# This version is labeled as "Updated 6/16" on the thread, fetched on May 18, 2017

# #Event Study
# 
# An event study is typically used to analyze the impact of an event/piece of information on a security. This notebook provides tools to visualize the average price movement a security experiences around an event. Specifically, it looks at:
# 
# - Simple cumulative price returns movement around the event date
# - Abnormal cumulative price returns movement around the event date
# - Error bars for the two above charts
# 
# For more information on event studies view: http://www.investopedia.com/terms/e/eventstudy.asp.
# 
# ## Directions for use:
# 
# 1. Run cells necessary for module imports and function definitions
# 2. Import events dataset of choice and call `run_event_study` on dataset (Last Cell of Notebook)

# In[4]:

from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import timedelta
from odo import odo
import scipy
import math

from quantopian.pipeline import Pipeline
from quantopian.research import run_pipeline
from quantopian.pipeline.classifiers.morningstar import Sector
from quantopian.pipeline.data import morningstar as mstar
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import CustomFactor, AverageDollarVolume, SimpleMovingAverage
from quantopian.pipeline.filters.morningstar import IsPrimaryShare


# In[5]:

def filter_universe(min_price = 0., min_volume = 0.):  
    """
    Computes a security universe based on nine different filters:

    1. The security is common stock
    2 & 3. It is not limited partnership - name and database check
    4. The database has fundamental data on this stock
    5. Not over the counter
    6. Not when issued
    7. Not depository receipts
    8. Is Primary share
    9. Has high dollar volume
    
    Returns
    -------
    high_volume_tradable - zipline.pipeline.factor.Rank
        A ranked AverageDollarVolume factor that's filtered on the nine criteria
    """
    common_stock = mstar.share_class_reference.security_type.latest.eq('ST00000001')
    not_lp_name = ~mstar.company_reference.standard_name.latest.matches('.* L[\\. ]?P\.?$')
    not_lp_balance_sheet = mstar.balance_sheet.limited_partnership.latest.isnull()
    have_data = mstar.valuation.market_cap.latest.notnull()
    not_otc = ~mstar.share_class_reference.exchange_id.latest.startswith('OTC')
    not_wi = ~mstar.share_class_reference.symbol.latest.endswith('.WI')
    not_depository = ~mstar.share_class_reference.is_depositary_receipt.latest
    primary_share = IsPrimaryShare()
    
    # Combine the above filters.
    tradable_filter = (common_stock & not_lp_name & not_lp_balance_sheet &
                       have_data & not_otc & not_wi & not_depository & primary_share)

    price = SimpleMovingAverage(inputs=[USEquityPricing.close],
                                window_length=252, mask=tradable_filter)
    volume = SimpleMovingAverage(inputs=[USEquityPricing.volume],
                                 window_length=252, mask=tradable_filter)
    full_filter = tradable_filter & (price >= min_price) & (volume >= min_volume)
    
    high_volume_tradable = AverageDollarVolume(
            window_length=252,
            mask=full_filter
        ).rank(ascending=False)
    return high_volume_tradable

class SidFactor(CustomFactor):
    """
    Workaround to screen by sids in pipeline
    
    Credit: Luca
    """
    inputs = []  
    window_length = 1  
    sids = []

    def compute(self, today, assets, out):
        out[:] = np.in1d(assets, self.sids)

def get_liquid_universe_of_stocks(start_date, end_date, top_liquid=500):    
    """
    Gets the top X number of securities based on the criteria defined in
    `filter_universe`
    
    Parameters
    ----------
    start_date : string or pd.datetime
        Starting date for universe computation.
    end_date : string or pd.datetime
        End date for universe computation.
    top_liquid : int, optional
        Limit universe to the top N most liquid names in time period.
        Based on 21 day AverageDollarVolume
        
    Returns
    -------
    security_universe : list
        List of securities that match the universe criteria
    """
    pipe = Pipeline()
    pipe.add(AverageDollarVolume(window_length=1), 'liquidity')
    pipe.set_screen((filter_universe() < top_liquid))
    data = run_pipeline(pipe, start_date=start_date, end_date=end_date)

    security_universe = data.index.levels[1].unique().tolist()
    return security_universe


# In[6]:

def get_cum_returns(prices, sid, date, days_before, days_after, benchmark_sid):
    """
    Calculates cumulative and abnormal returns for the sid & benchmark
    
    Parameters
    ----------
    prices : pd.DataFrame
        Pricing history DataFrame obtained from `get_pricing`. Index should
        be the datetime index and sids should be columns.
    sid : int or zipline.assets._assets.Equity object
        Security that returns are being calculated for.
    date : datetime object
        Date that will be used as t=0 for cumulative return calcuations. All
        returns will be calculated around this date.
    days_before, days_after : int
        Days before/after to be used to calculate returns for.
    benchmark :  int or zipline.assets._assets.Equity object
    
    Returns
    -------
    sid_returns : pd.Series
        Cumulative returns time series from days_before ~ days_after from date
        for sid
    benchmark_returns : pd.Series
        Cumulative returns time series for benchmark sid
    abnormal_returns : pd.Series
        Abnomral cumulative returns time series for sid compared against benchmark
    """

    day_zero_index = prices.index.searchsorted(date)
    starting_index = max(day_zero_index - days_before, 0)
    ending_index   = min(day_zero_index + days_after + 1, len(prices.index) - 1)

    if starting_index < 0 or ending_index >= len(prices.index):
        return None
    
    if sid == benchmark_sid:
        temp_price = prices.iloc[starting_index:ending_index,:].loc[:,[sid]]
    else:
        temp_price = prices.iloc[starting_index:ending_index,:].loc[:,[sid, benchmark_sid]]
            
    beta = calc_beta(sid, benchmark_sid, temp_price)
    if beta is None:
        return
    
    daily_ret = temp_price.pct_change().fillna(0)
    
    daily_ret['abnormal_returns'] = daily_ret[sid] - beta*daily_ret[benchmark_sid]
    
    cum_returns = (daily_ret + 1).cumprod() - 1
    
    try:
        # If there's not enough data for event study,
        # return None
        cum_returns.index = range(starting_index - day_zero_index,
                                  ending_index - day_zero_index)
    except:
        return None
    
    sid_returns      = cum_returns[sid] - cum_returns[sid].ix[0]
    bench_returns    = cum_returns[benchmark_sid] - cum_returns[benchmark_sid].ix[0]
    abnormal_returns = cum_returns['abnormal_returns'] - cum_returns['abnormal_returns'].ix[0]
    
    return sid_returns, bench_returns, abnormal_returns

def calc_beta(sid, benchmark, price_history):
    """
    Calculate beta amounts for each security
    
    Parameters
    ----------
    sid : int or zipline.assets._assets.Equity object
        Security that beta is being calculated for.
    benchmark : int or zipline.assets._assets.Equity object
        Benchmark that will be used to determine beta against
    price_history: pd.DataFrame
        DataFrame that contains pricing history for benchmark and
        sid. Index is a datetimeindex and columns are sids. Should 
        already be truncated for date_window used to calculate beta.
        
    Returns
    -------
    beta : float
        Beta of security against benchmark calculated over the time
        window contained in price_history
    """
    if sid == benchmark:
        return 1.0
    
    stock_prices = price_history[sid].pct_change().dropna()
    bench_prices = price_history[benchmark].pct_change().dropna()
    aligned_prices = bench_prices.align(stock_prices,join='inner')
    bench_prices = aligned_prices[0]
    stock_prices = aligned_prices[1]
    bench_prices = np.array( bench_prices.values )
    stock_prices = np.array( stock_prices.values )
    bench_prices = np.reshape(bench_prices,len(bench_prices))
    stock_prices = np.reshape(stock_prices,len(stock_prices))
    if len(stock_prices) == 0:
        return None
    regr_results = scipy.stats.linregress(y=stock_prices, x=bench_prices) 
    beta = regr_results[0]  
    p_value = regr_results[3]
    if p_value > 0.05:
        beta = 0.
    return beta  


# In[7]:

def define_xticks(days_before, days_after):
    """
    Defines a neat xtick label axis on multipes of 2 using X days before
    and X days after.
    
    Parameters
    ----------
    days_before : int
        Positive integer detailing the numbers of days before event date
    days_after : int
        Postiive integer detailing the number of days after an event date
        
    Returns
    -------
    list : List of integers on multiples of 2 from [-days_before ~ days_after)
    """
    day_numbers = [i for i in range(-days_before+1, days_after)]
    xticks = [d for d in day_numbers if d%2 == 0]
    return xticks

def plot_distribution_of_events(event_data, date_column, start_date, end_date):
    """
    Plots the distribution of events
    
    Parameters
    ----------
    event_data : pd.DataFrame
        DataFrame that contains the events data with date and sid columns as
        a minimum. See interactive tutorials on quantopian.com/data
    date_column : String
        String that labels the date column to be used for the event. e.g. `asof_date`
    start_date, end_date : Datetime
        Start and end date to be used for the cutoff for the distribution plots
    """
    event_data = event_data[(event_data[date_column] > start_date) &
                            (event_data[date_column] < end_date)]
    s = pd.Series(event_data[date_column])
    
    sns.set_palette('coolwarm')
    s.groupby([s.dt.year, s.dt.month]).count().plot(kind="bar", grid=False,
                                                    color=sns.color_palette())
    plt.title("Distribution of events in time")
    plt.ylabel("Number of event")
    plt.xlabel("Date")
    plt.show()
    
    
def plot_cumulative_returns(cumulative_returns, days_before, days_after):
    """
    Plots a cumulative return chart
    
    Parameters
    ----------
    cumulative_returns : pd.series
        Series that contains the cumulative returns time series from
        days_before ~ days_after from date for sid. See `get_cum_returns
    days_before, days_after : Datetime
        Positive integer detailing the numbers of days before/after event date
    """
    xticks = define_xticks(days_before, days_after)
    cumulative_returns.plot(xticks=xticks)
        
    plt.grid(b=None, which=u'major', axis=u'y')
    plt.title("Cumulative Return before and after event")
    plt.xlabel("Window Length (t)")
    plt.ylabel("Cumulative Return (r)")
    plt.legend(["N=%s" % cumulative_returns.name])
    plt.show()

def plot_cumulative_returns_with_error_bars(cumulative_returns, returns_with_error,
                                            days_before, days_after, abnormal=False):
    """
    Plots a cumulative return chart with error bars. Can choose between abnormal returns
    and simple returns
    
    Parameters
    ----------
    cumulative_returns : pd.Series
        Series that contains the cumulative returns time series from
        days_before ~ days_after from date for sid. See `get_cum_returns
    returns_with_error: pd.Series
        Series that contains the standard deviation of returns passed in through
        `cumulative_returns`. See `get_returns`
    days_before, days_after : Datetime
        Positive integer detailing the numbers of days before/after event date
    abnormal : Boolean, optional
        If True, will plot labels indicating an abnormal returns chart
    """
    xticks = define_xticks(days_before, days_after)
    returns_with_error.ix[:-1] = 0
    plt.errorbar(cumulative_returns.index, cumulative_returns, xerr=0, yerr=returns_with_error)
    plt.grid(b=None, which=u'major', axis=u'y')
    if abnormal:
        plt.title("Cumulative Abnormal Return before and after event with error")
    else:
        plt.title("Cumulative Return before and after event with error")
    plt.xlabel("Window Length (t)")
    plt.ylabel("Cumulative Return (r)")
    plt.legend()
    plt.show()
    
def plot_cumulative_returns_against_benchmark(cumulative_returns,
                                              benchmark_returns,
                                              days_before, days_after):
    """
    Plots a cumulative return chart against the benchmark returns
    
    Parameters
    ----------
    cumulative_returns, benchmark_returns : pd.series
        Series that contains the cumulative returns time series from
        days_before ~ days_after from date for sid/benchmark. See `get_cum_returns`
    days_before, days_after : Datetime
        Positive integer detailing the numbers of days before/after event date
    """
    xticks = define_xticks(days_before, days_after)
    cumulative_returns.plot(xticks=xticks, label="Event")
    benchmark_returns.plot(xticks=xticks, label='Benchmark')
    
    plt.title("Comparing the benchmark's average returns around that time to the event")
    plt.ylabel("% Cumulative Return")
    plt.xlabel("Time Window")
    plt.legend(["Event", 'Benchmark'])
    plt.grid(b=None, which=u'major', axis=u'y')
    plt.show()
    
def plot_cumulative_abnormal_returns(cumulative_returns,
                                     abnormal_returns,
                                     days_before, days_after):
    """
    Plots a cumulative return chart against the abnormal returns
    
    Parameters
    ----------
    cumulative_returns, abnormal_returns : pd.series
        Series that contains the cumulative returns time series against abnormal returns
        from days_before ~ days_after from date for sid. See `get_cum_returns`
    days_before, days_after : Datetime
        Positive integer detailing the numbers of days before/after event date
    """
    xticks = define_xticks(days_before, days_after)
    abnormal_returns.plot(xticks=xticks, label="Abnormal Average Cumulative")
    cumulative_returns.plot(xticks=xticks, label="Simple Average Cumulative")
    
    plt.axhline(y=abnormal_returns.ix[0], linestyle='--', color='black', alpha=.3, label='Drift')
    plt.axhline(y=abnormal_returns.max(), linestyle='--', color='black', alpha=.3)
    plt.title("Cumulative Abnormal Returns versus Cumulative Returns")
    plt.ylabel("% Cumulative Return")
    plt.xlabel("Time Window")
    plt.grid(b=None, which=u'major', axis=u'y')
    plt.legend(["Abnormal Average Cumulative","Simple Average Cumulative", 'Drift'])
    plt.show()
    
def get_returns(event_data, benchmark, date_column, days_before, days_after,
                use_liquid_stocks=False, top_liquid=1000):
    """
    Calculates cumulative returns, benchmark returns, abnormal returns, and
    volatility for cumulative and abnomral returns
    
    Parameters
    ----------
    event_data : pd.DataFrame
        DataFrame that contains the events data with date and sid columns as
        a minimum. See interactive tutorials on quantopian.com/data
    benchmark : string, int, zipline.assets._assets.Equity object
        Security to be used as benchmark for returns calculations. See `get_returns`
    date_column : String
        String that labels the date column to be used for the event. e.g. `asof_date`
    days_before, days_after : Datetime
        Positive integer detailing the numbers of days before/after event date
    use_liquid_stocks : Boolean
        If set to True, it will filter out any securities found in `event_data`
        according to the filters found in `filter_universe`
    top_liquid : Int
        If use_liquid_stocks is True, top_liquid determines the top X amount of stocks
        to return ranked on liquidity
        
        
    Returns
    -------
    cumulative_returns, benchmark_returns, abnormal_returns
    returns_volatiliy, abnormal_returns_volatility : pd.Series
    valid_sids: list
        Used to graph distribution of events (in case of use_liquid_stocks flag)
    """
    cumulative_returns = []
    benchmark_returns = []
    abnormal_returns = []
    valid_sids = []
    liquid_stocks = None
    
    print "Running Event Study"
    for i, row in event_data[['sid', date_column]].iterrows():
        sid, date = row
        
        # Getting 10 extra days of data just to be sure
        extra_days_before = math.ceil(days_before * 365.0/252.0) + 10
        start_date = date - timedelta(days=extra_days_before)
        extra_days_after = math.ceil(days_after * 365.0/252.0) + 10
        end_date   = date + timedelta(days=extra_days_after)

        if use_liquid_stocks:
            if liquid_stocks is None:
                liquid_stocks = get_liquid_universe_of_stocks(date, date, top_liquid=top_liquid)
            if sid not in liquid_stocks:
                continue
                
        valid_sids.append(sid)

        # duplicated columns would break get_cum_returns
        pr_sids = set([sid, benchmark])
        prices = get_pricing(pr_sids, start_date=start_date,
                             end_date=end_date, fields='open_price')
        prices = prices.shift(-1)
        if date in prices.index:
            results = get_cum_returns(prices, sid, date, days_before, days_after, benchmark)
            if results is None:
                print "Discarding event for %s on %s" % (symbols(sid),date)
                continue
            sid_returns, b_returns, ab_returns = results
            cumulative_returns.append(sid_returns)
            benchmark_returns.append(b_returns)
            abnormal_returns.append(ab_returns)
            
    sample_size = len(cumulative_returns)
    returns_volatility          = pd.concat(cumulative_returns, axis=1).std(axis=1)
    abnormal_returns_volatility = pd.concat(abnormal_returns,   axis=1).std(axis=1)
    benchmark_returns           = pd.concat(benchmark_returns,  axis=1).mean(axis=1)
    abnormal_returns            = pd.concat(abnormal_returns,   axis=1).mean(axis=1)
    cumulative_returns          = pd.concat(cumulative_returns, axis=1).mean(axis=1)
    cumulative_returns.name = sample_size
        
    return (cumulative_returns, benchmark_returns, abnormal_returns,
            returns_volatility, abnormal_returns_volatility, valid_sids)


# In[8]:

def run_event_study(event_data, date_column='asof_date',
                    start_date='2007-01-01', end_date='2014-01-01',
                    benchmark=None, days_before=10, days_after=10, top_liquid=500,
                    use_liquid_stocks=True):
    """
    Calculates simple & cumulative returns for events and plots stock price movement
    before and after the event date.
    
    Parameters
    ----------
    event_data : pd.DataFrame
        DataFrame that contains the events data with date and sid columns as
        a minimum. See interactive tutorials on quantopian.com/data
    date_column : String
        String that labels the date column to be used for the event. e.g. `asof_date`
    start_date, end_date : Datetime
        Start and end date to be used for the cutoff for the evenet study
    benchmark : int or zipline.assets._assets.Equity object
        Security to be used as benchmark for returns calculations. See `get_returns`
    days_before, days_after : int
        Days before/after to be used to calculate returns for.
    top_liquid : Int
        If use_liquid_stocks is True, top_liquid determines the top X amount of stocks
        to return ranked on liquidity
    use_liquid_stocks : Boolean
        If set to True, it will filter out any securities found in `event_data`
        according to the filters found in `filter_universe`
    """
    if date_column not in event_data or not isinstance(event_data, pd.DataFrame) or 'sid' not in event_data:
        raise KeyError("event_data not properly formatted for event study. Please make sure "                        "date_column and 'sid' are both present in the DataFrame")

    if isinstance(benchmark, str):
        raise TypeError("Benchmark must be an equity object. Please use symbols('ticker') to"                         "set your benchmark")
        
    if benchmark is None:
        benchmark = symbols('SPY')
        
    print "Formatting Data"
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    event_data = event_data[(event_data[date_column] > start_date) &
                            (event_data[date_column] < end_date)]
    event_data.sid = event_data.sid.apply(lambda x: int(x))
    
    print "Getting Plots"
    cumulative_returns, benchmark_returns, abnormal_returns, returns_volatility,         abnormal_returns_volatility, valid_sids = get_returns(event_data, benchmark, date_column,
                                                              days_before, days_after,
                                                              use_liquid_stocks=use_liquid_stocks,
                                                              top_liquid=top_liquid)
    event_data = event_data[event_data.sid.isin(valid_sids)]
    plot_distribution_of_events(event_data, date_column, start_date, end_date)

    plot_cumulative_returns(cumulative_returns, days_before, days_after)
    
    plot_cumulative_returns_against_benchmark(cumulative_returns, benchmark_returns,
                                              days_before, days_after)
    
    plot_cumulative_abnormal_returns(cumulative_returns, abnormal_returns,
                                     days_before, days_after)
    
    plot_cumulative_returns_with_error_bars(cumulative_returns, returns_volatility,
                                            days_before, days_after)
    
    plot_cumulative_returns_with_error_bars(cumulative_returns, abnormal_returns_volatility,
                                            days_before, days_after, abnormal=True)


# # Run Event Study from 2013 ~ 2014

# In[10]:

from quantopian.interactive.data.zacks import earnings_surprises

# [2013, 2014)
#years = range(2013, 2015)
years = 2015,

# Negative earnings surprises of -50% or more
# Break it up into years so we can actually load in all the data
for year in years:
    start = '%s-01-01' % year
    end =  '%s-12-31' % year
    temp_data = earnings_surprises[earnings_surprises.eps_pct_diff_surp < -.50]
    temp_data = temp_data[temp_data.asof_date >= pd.to_datetime(start)]
    temp_data = temp_data[temp_data.asof_date <= pd.to_datetime(end)]
    df = odo(temp_data, pd.DataFrame)
    print "Running event study for %s" % year
    run_event_study(df, start_date=start, end_date=end, use_liquid_stocks=False, top_liquid=500)
