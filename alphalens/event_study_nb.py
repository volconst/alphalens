
# coding: utf-8

# This is an IPython notebook from https://www.quantopian.com/posts/event-study-tearsheet
# Added to git as .py for easier tracking of changes.
# Not runnable as .py, copy in quantopian.com research notebook to run.

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


class PerfStats:
    count = 0
    seconds = 0.
    def __repr__(self):
        return 'seconds: %.2f count: %d'%(self.seconds, self.count)
import time
def timer(some_function):
    #timing decorator

    def wrapper(*args, **kwargs):
        t1 = time.time()
        res = some_function(*args, **kwargs)
        timer.stats[some_function.func_name].seconds += time.time() - t1
        timer.stats[some_function.func_name].count += 1
        return res
    return wrapper
from collections import defaultdict
timer.stats = defaultdict(PerfStats)

# In[6]:

@timer
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
    #print 'day_zero_index', day_zero_index
    starting_index = max(day_zero_index - days_before, 0)
    ending_index   = min(day_zero_index + days_after + 1, len(prices.index) - 1)

    if starting_index < 0 or ending_index >= len(prices.index):
        assert False #is this possible
        return None
    
#     if sid == benchmark_sid:
#         temp_price = prices.iloc[starting_index:ending_index,:].loc[:,[sid]]
#     else:
#        temp_price = prices.iloc[starting_index:ending_index,:].loc[:,[sid, benchmark_sid]]
    temp_price = prices.iloc[starting_index:ending_index][[sid, benchmark_sid]]
            
    beta = calc_beta(sid, benchmark_sid, temp_price)
    if beta is None:
        #print 'beta is None'
        return
    
    daily_ret = temp_price.pct_change().fillna(0)
    
    daily_ret['abnormal_returns'] = daily_ret[sid] - beta*daily_ret[benchmark_sid]
    
    cum_returns = (daily_ret + 1).cumprod() - 1
    
    try:
        # If there's not enough data for event study,
        # return None
        cum_returns.index = range(starting_index - day_zero_index,
                                  ending_index - day_zero_index)
    except e:
        print 'exception', e
        return None

    cum_returns = cum_returns - cum_returns.loc[0]  #aligns returns at day 0
    
    return cum_returns[sid], cum_returns[benchmark_sid], cum_returns['abnormal_returns']


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

#     #print 'price_history[sid]', price_history[sid]
#     stock_prices = price_history[sid].pct_change().dropna()
#     #print 'stock_prices', stock_prices
#     bench_prices = price_history[benchmark].pct_change().dropna()
#     aligned_prices = bench_prices.align(stock_prices,join='inner')
#     if len(stock_prices) != len(aligned_prices[1]):
#         print 'aligned change', stock_prices, aligned_prices[1]
#     #print 'aligned_prices', aligned_prices
#     bench_prices = aligned_prices[0]
#     stock_prices = aligned_prices[1]
#     bench_prices = np.array( bench_prices.values )
#     stock_prices = np.array( stock_prices.values )
#     bench_prices = np.reshape(bench_prices,len(bench_prices))
#     stock_prices = np.reshape(stock_prices,len(stock_prices))
#     if len(stock_prices) == 0:
#         return None
#     regr_results = scipy.stats.linregress(y=stock_prices, x=bench_prices) 

    #TODO: the order of dropna and pct_change matters. Maybe not - pct_change skips nan (equivalent to fillna(ffill)) when looking at previous return
    #price_history = price_history[[sid,benchmark]].dropna().pct_change()[1:]
    assert len(price_history.columns) == 2
    #price_history = price_history[[sid, benchmark]].pct_change().dropna()
    price_history = price_history.pct_change().dropna()
    if price_history.empty:
        return None

    regr_results2 = scipy.stats.linregress(y=price_history[sid], x=price_history[benchmark])
    regr_results = regr_results2
#     assert (stock_prices - price_history[sid].pct_change()[1:].values == 0).all()
#     assert (bench_prices - price_history[benchmark].pct_change()[1:].values == 0).all(), '%s %s'%(bench_prices, price_history[benchmark].pct_change()[1:].values)
#    assert regr_results == regr_results2
    #, 'sid %s sid prices %s\nbench prices %s\naligned %s\nreg %s\nreg2 %s'%\
    #    (sid, price_history[sid].pct_change()[1:].values, price_history[benchmark].pct_change()[1:].values, stock_prices, regr_results, regr_results2)
    beta = regr_results[0]  
    p_value = regr_results[3]
    if p_value > 0.05:
        beta = 0.
    return beta  

@timer
def calc_beta_from_pct(sid_pct_change, benchmark_pct_change):
    if sid_pct_change.name == benchmark_pct_change.name:
        return 1.0
    mask = sid_pct_change.notnull() & benchmark_pct_change.notnull()
    if not mask.any():
        return None

    regr_results = scipy.stats.linregress(y=sid_pct_change[mask], x=benchmark_pct_change[mask])
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
    
weeks_to_fetch = 3

@timer
def get_returns(event_data, benchmark, date_column, days_before, days_after,
                use_liquid_stocks=False, top_liquid=1000, get_pricing_func=get_pricing):
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

    print "Running Event Study"
    if use_liquid_stocks:
        liquid_stocks = get_liquid_universe_of_stocks(start_date, start_date, top_liquid=top_liquid)

    # Getting 10 extra days of data just to be sure
    extra_days_before = timedelta(days=math.ceil(days_before * 365.0/252.0) + 10)
    extra_days_after = timedelta(days=math.ceil(days_after * 365.0/252.0) + 10)
    benchmark_prices = get_pricing_func(benchmark, start_date=event_data[date_column].min()-extra_days_before,
                                   end_date=event_data[date_column].max()+extra_days_after, fields='open_price')
    for _, week_group in event_data.groupby([event_data[date_column].dt.year,
                                             event_data[date_column].dt.weekofyear//weeks_to_fetch]):
        
        start_date = week_group[date_column].min() - extra_days_before
        end_date   = week_group[date_column].max() + extra_days_after

        # duplicated columns would break get_cum_returns - how?
        pr_sids = set(week_group.sid)
        if use_liquid_stocks:
            pr_sids.intersection_update(liquid_stocks)

        #print 'pr_sids', [sid.symbol for sid in pr_sids]
        print 'get_pricing(%d sids, %s, %s)'%(len(pr_sids), start_date.date(), end_date.date())
        prices = get_pricing_func(pr_sids, start_date=start_date,
                             end_date=end_date, fields='open_price')
        prices[benchmark_prices.name] = benchmark_prices
        #prices = prices.shift(-1)  #why?
        global temp_price, pct_change, daily_ret, cum_returns, benchmark_cum, date_group

        for dt, date_group in week_group.groupby(date_column):

            day_zero_index = prices.index.searchsorted(dt)
            if prices.index[day_zero_index].date() != dt.date():
                print 'Event date %s without price, discard sids %s' % (dt.strftime('%a %Y-%m-%d'), date_group.sid.values)
                #TODO: maybe use previous date price
                continue

            #print 'day_zero_index', day_zero_index
            starting_index = day_zero_index - days_before
            ending_index   = day_zero_index + days_after + 1

            if starting_index < 0:
                print 'Prices did not contain enough days_before, discard sids'%date_group.sid
                continue
            if ending_index >= len(prices):
                print 'Prices did not contain enough days_after, discard sids'%date_group.sid
                continue
            temp_price = prices.iloc[starting_index:ending_index]
            temp_price.index = range(-days_before, days_after+1)
            #pct_change = temp_price.pct_change()
            benchmark_pct_change = temp_price[benchmark].pct_change()
            benchmark_pct_change_fill = benchmark_pct_change.fillna(0)

            #limit to current day symbols only
            date_group_prices = temp_price.loc[:,temp_price.columns.isin(date_group.sid)]
            all_nan_symbols = date_group_prices.isnull().all()
            if all_nan_symbols.any():
                print 'Discarded symbols with all Nan quotes', \
                    ', '.join(['%s %d'%(s.symbol, s.sid) for s in all_nan_symbols[all_nan_symbols].index])

            for sid in all_nan_symbols[~all_nan_symbols].index:

                if use_liquid_stocks and sid not in liquid_stocks:
                    continue
                valid_sids.append(sid)

                #print 'get_pricing', pr_sids
                sid_pct_change = temp_price[sid].pct_change()
                beta = calc_beta_from_pct(sid_pct_change, benchmark_pct_change)
                if beta is None:
                    print 'beta is None. Discard sid %s'%sid
                    continue
                #daily_ret = pct_change[[sid, benchmark]].fillna(0)
                #daily_ret['abnormal_returns'] = daily_ret[sid] - beta*benchmark_pct_change
                abnormal_pct_change = sid_pct_change.fillna(0) - beta*benchmark_pct_change_fill
                abnormal_cumprod = (abnormal_pct_change + 1).cumprod()
                #cum_returns = (daily_ret + 1).cumprod() - 1
                # day alignment is incorrect - must divide by loc[0] instead of just subtracting
                # so the changes of other days are measured in day zero percents
                ##cum_returns = cum_returns / cum_returns.loc[0] - 1
                #cum_returns = cum_returns - cum_returns.loc[0]  #aligns returns at day 0

                #benchmark_cum = temp_price[benchmark]/temp_price.loc[0,benchmark] - 1
                def calc_cumul(prices):
                    #incorrect, but compatible
                    #return (prices-prices.loc[0])/prices.iloc[0]
                    return prices/prices.loc[0] - 1
                benchmark_cum = calc_cumul(temp_price[benchmark])
                sid_cum = calc_cumul(temp_price[sid])
                abnormal_cum = calc_cumul(abnormal_cumprod)
                #assert (cum_returns[benchmark] - benchmark_cum).abs().max() < 1e-10, 'alt cum method verify %s'%(cum_returns[benchmark] - benchmark_cum)
                #assert (cum_returns[sid] - sid_cum).abs().fillna(0).max() < 1e-10, 'alt cum method verify %s'%(cum_returns[sid] - sid_cum)

                cumulative_returns.append(sid_cum)
                benchmark_returns.append(benchmark_cum)
                abnormal_returns.append(abnormal_cum)
            
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

@timer
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

#test with single event
def get_pricing_custom(pr_sids, start_date, end_date, fields):
    print 'get_pricing_custom', pr_sids, start_date, end_date, fields
    global prices
    prices = pd.DataFrame({'date': pd.date_range(start_date, end_date)})
    single_sid = isinstance(pr_sids, int)
    if single_sid:
        pr_sids = pr_sids,
    for sid in pr_sids:
        prices[sid] = 1 #+ np.array(range(len(prices)))/10
    prices.set_index('date', inplace=True)
    prices.loc['2013-2-1'] = 1
    prices.loc['2013-2-2'] = 1.2
    return prices[pr_sids[0]] if single_sid else prices

df = pd.DataFrame([[pd.to_datetime('2013-2-1'),1]], columns=['asof_date', 'sid'])
print 'events:\n', df
# run_event_study(df, date_column='asof_date', start_date='2013-1-31', end_date='2013-2-3',
#                     benchmark=1, days_before=0, days_after=1, top_liquid=500,
#                     use_liquid_stocks=False)
get_returns(df, 1, 'asof_date', days_before=0, days_after=1, use_liquid_stocks=False,
            get_pricing_func=get_pricing_custom)[0]

#test with single event days_before percents
def get_pricing_days_before(pr_sids, start_date, end_date, fields):
    print 'get_pricing_custom', pr_sids, start_date, end_date, fields
    prices = pd.DataFrame({'date': pd.date_range(start_date, end_date)})
    single_sid = isinstance(pr_sids, int)
    if single_sid:
        pr_sids = pr_sids,
    for sid in pr_sids:
        prices[sid] = 1 #+ np.array(range(len(prices)))/10
    prices.set_index('date', inplace=True)
    prices.loc['2013-2-1'] = 0.8
    prices.loc['2013-2-2'] = 1.
    return prices[pr_sids[0]] if single_sid else prices

df = pd.DataFrame([[pd.to_datetime('2013-2-2'),1]], columns=['asof_date', 'sid'])
print 'events:\n', df
cumul_rets = get_returns(df, 1, 'asof_date', days_before=1, days_after=0, use_liquid_stocks=False,
            get_pricing_func=get_pricing_days_before)[0]
print cumul_rets
assert math.fabs(cumul_rets.loc[-1] - -0.2) < 1e-10, 'expected -0.2 = (1-0.8)/1'


# # Run Event Study from 2013 ~ 2014

# In[10]:

from quantopian.interactive.data.zacks import earnings_surprises

# [2013, 2014)
#years = range(2013, 2015)
years = 2014,
import time

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
    start_time = time.time()
    run_event_study(df, start_date=start, end_date=end, use_liquid_stocks=False, top_liquid=500)
    print 'took', time.time()-start_time, 'seconds'
