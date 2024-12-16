import pandas as pd
import numpy as np
from datetime import timedelta


class TradingTimeDelta:
    """
    A class to calculate the difference in time (delta T) between two timestamps,
    considering only trading hours on trading days.
    
    Trading Hours:
        - Days: Monday to Friday
        - Hours: 9:30 AM to 4:00 PM
    """
    
    def __init__(self, start_time, end_time):
        """
        Initializes the TradingTimeDelta object with start and end times.
        
        Parameters:
            start_time (str, int, float, pd.Timestamp): The start time.
            end_time (str, int, float, pd.Timestamp): The end time.
            
        Raises:
            ValueError: If start_time is after end_time.
        """
        # convert inputs to pandas Timestamps
        self.start_time = pd.to_datetime(start_time)
        self.end_time = pd.to_datetime(end_time)
        
        if self.start_time > self.end_time:
            raise ValueError("start_time must be less than or equal to end_time.")
        
        # trading hours
        self.trading_start_hour = 9
        self.trading_start_minute = 30
        self.trading_end_hour = 16
        self.trading_end_minute = 0

    def _is_trading_day(self, day):
        """
        Checks if a given day is a trading day (Monday to Friday).
        
        Parameters:
            day (pd.Timestamp): The day to check.
        
        Returns:
            bool: True if trading day, False otherwise.
        """
        return day.weekday() < 5  # Monday=0, Sunday=6

    def _get_trading_period(self, day):
        """
        Returns the trading start and end times for a given day.
        
        Parameters:
            day (pd.Timestamp): The day for which to get trading hours.
        
        Returns:
            tuple: (trading_start, trading_end) as pd.Timestamp objects.
        """
        trading_start = pd.Timestamp(
            year=day.year, month=day.month, day=day.day,
            hour=self.trading_start_hour, minute=self.trading_start_minute
        )
        trading_end = pd.Timestamp(
            year=day.year, month=day.month, day=day.day,
            hour=self.trading_end_hour, minute=self.trading_end_minute
        )
        return trading_start, trading_end

    def _calculate_trading_seconds(self):
        """
        Calculates the total trading seconds between start_time and end_time,
        and checks if any trading day ends exactly at trading_end.
        
        Returns:
            tuple: (total_trading_seconds, trading_end_included)
        """
        total_trading_seconds = 0
        trading_end_included = False
        
        # date range from start to end date
        date_range = pd.date_range(start=self.start_time.date(), end=self.end_time.date(), freq='D')
        
        for day in date_range:
            if not self._is_trading_day(day):
                continue  # skip weekends
            
            trading_start, trading_end = self._get_trading_period(day)
            
            # find overlap between [trading_start, trading_end] and [start_time, end_time]
            current_start = max(self.start_time, trading_start)
            current_end = min(self.end_time, trading_end)
            
            if current_start >= current_end:
                continue  
            
            # get seconds for this trading day
            trading_seconds = (current_end - current_start).total_seconds()
            total_trading_seconds += trading_seconds
            
            # check if trading_end is included in the overlap
            if current_end == trading_end:
                trading_end_included = True
        
        return total_trading_seconds, trading_end_included

    def get_delta_minutes(self):
        """
        Returns the delta T in minutes during trading hours.
        
        Returns:
            float: Delta T in minutes.
        """
        total_seconds, _ = self._calculate_trading_seconds()
        delta_minutes = total_seconds / 60
        return delta_minutes

    def get_delta_hours(self):
        """
        Returns the delta T in hours during trading hours.
        Adds 0.5 hours if any trading day in the range includes the trading end time.
        This is because yfinance has the last hourly period start at 3:30 PM, and thus is 
        considered as one full hourly point, while only lasting 30 minutes.
        
        Returns:
            float: Delta T in hours.
        """
        total_seconds, trading_end_included = self._calculate_trading_seconds()
        delta_hours = total_seconds / 3600
        if trading_end_included:
            delta_hours += 0.5
        return delta_hours

    def get_delta_days(self):
        """
        Returns the delta T in days during trading hours.
        A trading day is considered as 6.5 hours.
        
        Returns:
            float: Delta T in days.
        """
        total_seconds, _ = self._calculate_trading_seconds()
        delta_days = total_seconds / (6.5 * 3600)  # 6.5 trading hours per day
        return delta_days
    
    def get_delta_seconds(self):
        return self.get_delta_t(unit='seconds')

    def get_delta_t(self, unit='seconds'):
        """
        General method to get delta T in specified units.
        
        Parameters:
            unit (str): The unit for delta T ('seconds', 'minutes', 'hours', 'days').
        
        Returns:
            float: Delta T in the specified unit.
        
        Raises:
            ValueError: If an unsupported unit is specified.
        """
        total_seconds, trading_end_included = self._calculate_trading_seconds()
        
        if unit == 'seconds':
            return total_seconds
        elif unit == 'minutes':
            return total_seconds / 60
        elif unit == 'hours':
            delta_hours = total_seconds / 3600
            if trading_end_included:
                delta_hours += 0.5
            return delta_hours
        elif unit == 'days':
            return total_seconds / (6.5 * 3600)
        else:
            raise ValueError("Unsupported unit. Choose from 'seconds', 'minutes', 'hours', 'days'.")
        
    @staticmethod
    def get_next_trading_time(current_time: pd.Timestamp) -> pd.Timestamp:
        """
        Returns the next trading time after the given time.
        
        Parameters:
            current_time (pd.Timestamp): The current time.
        
        Returns:
            pd.Timestamp: The next trading time.
        """
        # check if current time is a trading day
        if current_time.weekday() < 5:
            # check if current time is after trading hours
            if current_time.hour >= 16:
                # find the next trading day at 9:30 AM
                next_day = current_time + pd.Timedelta(days=1)
                next_trading_time = pd.Timestamp(
                    year=next_day.year, month=next_day.month, day=next_day.day,
                    hour=9, minute=30
                )
            elif current_time.hour < 9.5:
                # trading hours have not started yet
                next_trading_time = pd.Timestamp(
                    year=current_time.year, month=current_time.month, day=current_time.day,
                    hour=9, minute=30
                )
            else:
                next_trading_time = current_time

        else:
            # find next weekday at 9:30 AM
            if current_time.weekday() == 5:  # Saturday
                next_day = current_time + pd.Timedelta(days=2)
            else:  # Sunday
                next_day = current_time + pd.Timedelta(days=1)
            next_trading_time = pd.Timestamp(
                year=next_day.year, month=next_day.month, day=next_day.day,
                hour=9, minute=30
            )

        return next_trading_time
    
    @staticmethod
    def add_trading_time(ts: pd.Timestamp, increment: timedelta = timedelta(hours=1)) -> pd.Timestamp:
        """
        Add one increment of trading time. If increment goes past 16:00, move to next day 9:30.

        Parameters:
            ts (pd.Timestamp): The current timestamp.
            increment (timedelta): The increment to add (default: 1 hour).
        """
        new_ts = ts + increment
        trading_start = new_ts.replace(hour=9, minute=30, second=0, microsecond=0)
        trading_end = new_ts.replace(hour=16, minute=0, second=0, microsecond=0)

        # If new_ts goes beyond trading_end, jump to the next trading day 9:30
        if new_ts > trading_end:
            # Move to next valid trading day at 9:30
            return TradingTimeDelta.get_next_trading_time(new_ts)
        # if the increment crosses into weekend, fix that too
        if not TradingTimeDelta.get_next_trading_time(new_ts):
            return TradingTimeDelta.get_next_trading_time(new_ts)
        return new_ts

    def generate_trading_timestamps(start_time: pd.Timestamp, steps: int, increment: timedelta = timedelta(hours=1)) -> pd.DatetimeIndex:
        """
        Generate 'steps' trading timestamps starting from start_time,
        moving in 1-hour increments within trading hours (9:30-16:00).
        """
        current = TradingTimeDelta.get_next_trading_time(start_time)
        timestamps = [current]
        for _ in range(steps - 1):
            current = TradingTimeDelta.add_trading_time(current, increment)
            timestamps.append(current)
        return pd.DatetimeIndex(timestamps)

# ---------------------- Example Usage ----------------------

if __name__ == "__main__":
    # Test 1
    start = "2024-10-15 10:30:00"
    end = "2024-10-16 14:30:00"
    
    delta = TradingTimeDelta(start, end)
    print("Test 1:")
    print("Delta Minutes:", delta.get_delta_minutes())  # 630
    print("Delta Hours:", delta.get_delta_hours())      # 11
    print("Delta Days:", delta.get_delta_days())        # ~1.615

    # Test 2: Across a Weekend
    start = "2024-10-18 10:30:00"  # Friday
    end = "2024-10-21 14:30:00"    # Monday
    delta = TradingTimeDelta(start, end)
    print("Test 2:")
    print("Delta Minutes (Across Weekend):", delta.get_delta_minutes())  # should account only trading hours on Fri and Mon - same as Test 1
    print("Delta Hours (Across Weekend):", delta.get_delta_hours())        
    print("Delta Days (Across Weekend):", delta.get_delta_days())          

    # Test 3: Start and End on Same Day
    start = "2024-10-17 09:00:00"  # Wednesday before trading
    end = "2024-10-17 17:00:00"    # Wednesday after trading
    delta = TradingTimeDelta(start, end)
    print("Test 3:")
    print("Delta Minutes (Same Day):", delta.get_delta_minutes())  # trading hours: 9:30 to 16:00 => 6.5 hours => 390 minutes
    print("Delta Hours (Same Day):", delta.get_delta_hours())      # 6.5 + 0.5 = 7
    print("Delta Days (Same Day):", delta.get_delta_days())        # 6.5 / 6.5 = 1

    # Test 4: Start and End Outside Trading Hours
    start = "2024-10-17 07:00:00"  # Wednesday before trading
    end = "2024-10-17 08:00:00"    # Wednesday before trading
    delta = TradingTimeDelta(start, end)
    print("Test 4:")
    print("Delta Minutes (Outside Trading Hours):", delta.get_delta_minutes())  # Expected: 0
    print("Delta Hours (Outside Trading Hours):", delta.get_delta_hours())        # Expected: 0
    print("Delta Days (Outside Trading Hours):", delta.get_delta_days())          # Expected: 0

    # Test 5: Partial Trading Days
    start = "2024-10-17 15:30:00"  # Wednesday 3:30 PM
    end = "2024-10-17 16:30:00"    # Wednesday 4:30 PM
    delta = TradingTimeDelta(start, end)
    print("Test 5:")
    print("Delta Minutes (Partial Day):", delta.get_delta_minutes())  # 15:30 to 16:00 => 30 minutes
    print("Delta Hours (Partial Day):", delta.get_delta_hours())      # 0.5 + 0.5 = 1
    print("Delta Days (Partial Day):", delta.get_delta_days())        # 0.5 / 6.5 ≈ 0.077

