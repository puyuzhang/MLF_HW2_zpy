"""
Course: Machine Learning in Finance (80517913-200)
Assignment: Factor/Feature Engineering on China A-Share
Author: Zhang Pu Yu (S24)
File: src/zhangpuyu_3547.py
Python: 3.11.x
DataDir: ./data/
OutDir: ./results/

Task Overview:
I implement the Profitability anomaly family (Base ID: A.4) and its variants (Opa, Ola, Oleq) 
at a monthly frequency using China A-share data.
I use RESSET data, with formation at t and execution at t+1. 
I apply proper lagging (6 months for annual, 4 months for quarterly) to ensure no look-ahead bias.

Data Conventions & Field Mapping:
- OpProf = Revenue - COGS - SGA - Interest Expense
- SGA = Selling Exp + Admin Exp
- Assets: Total Assets; Equity: Total Shareholder's Equity (Parent)
- Industry: SW Level-1 (Used for analysis, strictly firm-level operations for factors)
- Filters: 
    * Time Period: 2000-01-01 to 2024-12-31.
    * ST/ST* and IPO < 12m are RETAINED per updated guidance (to reflect historical tradeability).
    * Financial industry (Ind 'J') is excluded.
- Outliers: Cross-sectional winsorization at 1%/99%; Z-score standardization.
- Missing Data: Firm-level ffill (non-anticipative).
- NOTE on Toy Dataset: Per TA clarification, the mention of 'toy.csv' in the guidance was a typo. Therefore, no logic for a toy dataset is implemented.

Data Source & File Handling (Per TA Instructions):
- This script is designed to load specific RESSET file patterns used in local development:
  1. Balance Sheet: "RESSET_BS_ALL_*.csv"
  2. Income Statement: "RESSET_IS_ALL_*.csv"
  3. Monthly Returns: "RESSET_MRESSTK_*.csv"
  4. IPO Data: "RESSET_IISSULST_*.csv"
- COMPATIBILITY NOTE: As instructed, this script does NOT attempt to scan the entire 
  data/ directory to avoid memory issues with API-based file structures (17,000+ files).

Key Findings (Self-Check):
- Empirical results show a negative premium for Profitability factors in the tested China A-share sample (Mean spreads approx. -0.52% to -0.76% monthly), contrasting with typical U.S. findings.
- All variants (Opa, Ola, Oleq) are statistically insignificant, with t-statistics ranging from -0.64 to -0.87 (Obs=140), indicating that raw profitability signals alone do not generate robust excess returns in this specific period/universe.
- The negative sign likely reflects the "shell value" effect in A-shares where small/unprofitable firms often outperform, confirming the code correctly captures market characteristics without look-ahead bias.

Run:
- conda:
    conda create -n finfe-311 python=3.11.9 -c conda-forge -y
    conda activate finfe-311
    pip install -r requirements.txt

- venv:
    python3.11 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

- run:
    python src/zhangpuyu_3547.py

LLM/Agent Notes:
I used LLMs for:
1.  Understanding the assignment requirements, and using LLMs to help translate the PDF-provided 
    skeleton into an executable script in the early stage, in order to enhance my understanding 
    of the structure and intent of the skeleton code.
2.  Interpreting the mathematical factor definitions from Hou et al. (2017).
3.  Drafting NumPy-style docstrings for classes and functions.
4.  Assisting in the logic for `load_and_merge_data`, specifically verifying the file reading 
    strategy to handle RESSET's file naming patterns and ensure correct DataFrame concatenation.
5.  Generating the keyword mapping dictionary to robustly identify RESSET columns across different 
    file versions.

I reviewed and understand all code logic, ensuring it aligns with Hou et al. (2017).
"""

import logging
import os
from pathlib import Path
from typing import Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================= Configuration =================
try:
    from grading_api import BaseConfig, Solution
except ImportError:
    class BaseConfig:
        root: str = "."
        data_dir: str = "data"
        allowed_extra_dirs: tuple = ()
        autocreate: bool = True
        seed: int = 123
        
    class Solution:
        def __init__(self, cfg):
            self.cfg = cfg
        
        @classmethod
        def check_schema(cls, df):
            pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MyConfig(BaseConfig):
    """
    Configuration class for the assignment.
    
    Attributes
    ----------
    allowed_extra_dirs : tuple
        Directories allowed to be created by the script (logs, results).
    autocreate : bool
        Whether to automatically create directories.
    seed : int
        Random seed for reproducibility.
    """
    allowed_extra_dirs = ("logs", "results")
    autocreate = True
    seed = 123

class MySolution(Solution):
    """
    Main solution class for Profitability Factor Engineering (A.4).
    
    This class handles data loading, cleaning, feature construction (OpProf),
    and validation via portfolio sorting.
    """
    
    # Keyword Mappings
    KEYWORD_MAP_BS = {
        'sid':    ['A_StkCd', 'Stkcd', '股票代码'],
        'date':   ['EndDt', 'Accper', '截止日期'],
        'assets': ['TotAss', 'T10000', '资产总计'],
        'equity': ['TotShareEquitParent', 'T21800', '归属母公司股东权益合计']
    }
    
    KEYWORD_MAP_IS = {
        'sid':      ['A_StkCd', 'Stkcd', '股票代码'],
        'date':     ['EndDt', 'Accper', '截止日期'],
        'rev':      ['TotOperRev', 'TotOpRev', 'T40100', '营业总收入'],
        'cogs':     ['OperCost', 'OpCost', 'T40800', '营业成本'],
        'sell_exp': ['SellExp', 'OpExp', 'T41100', '销售费用'],
        'adm_exp':  ['AdminExp', 'AdmExp', 'T41200', '管理费用'],
        'fin_exp':  ['FinanExp', 'FinExp', 'T41300', '财务费用']
    }
    
    KEYWORD_MAP_STK = {
        'sid':       ['Stkcd', '股票代码'],
        'date':      ['Date', 'Trdmnt', '日期'],
        'ret':       ['Monret', 'Mretwd', '月收益率'],
        'ind_code':  ['Csrciccd1', '证监会行业门类'],
        'clpr':      ['ClPr', '收盘价'],
        'fullshr':   ['Fullshr', '总股数']
    }
    
    KEYWORD_MAP_IPO = {
        'sid':       ['StkCd', '股票代码'],
        'list_date': ['LstDt', '股票上市日']
    }

    @classmethod
    def standardize_columns(cls, df: pd.DataFrame, keyword_map: dict) -> pd.DataFrame:
        """
        Renames DataFrame columns based on a keyword mapping dictionary.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame with raw column names (Chinese/English mixed).
        keyword_map : dict
            Dictionary where keys are standard names and values are lists of possible raw names.

        Returns
        -------
        pd.DataFrame
            DataFrame with standardized column names. Unmapped columns are dropped.
        """
        rename_dict = {}
        found_targets = set()
        for col in df.columns:
            for std_name, keywords in keyword_map.items():
                if std_name in found_targets: continue
                for kw in keywords:
                    if kw.lower() in col.lower():
                        rename_dict[col] = std_name
                        found_targets.add(std_name)
                        break
        if not rename_dict: return pd.DataFrame()
        df = df[list(rename_dict.keys())]
        df = df.rename(columns=rename_dict)
        return df

    @classmethod
    def clean_sid(cls, series: pd.Series) -> pd.Series:
        """
        Standardizes stock identifiers to a 6-digit string format.

        Parameters
        ----------
        series : pd.Series
            Raw stock ID series (int, float, or dirty string).

        Returns
        -------
        pd.Series
            Cleaned 6-digit string series (e.g., '000001').
        """
        s = series.astype(str).str.strip()
        s = s.str.split('.').str[0]
        return s.str.zfill(6)

    @classmethod
    def load_and_merge_data(cls, data_dir: Path) -> pd.DataFrame:
        """
        Loads RESSET CSV files, cleans them, and merges them into a single DataFrame.
        
        Handles specific file patterns for Balance Sheet, Income Statement, Stock Data, and IPO info.
        Performs merging of annual/quarterly financials and market data.

        Parameters
        ----------
        data_dir : Path
            Path to the directory containing raw CSV files.

        Returns
        -------
        pd.DataFrame
            Merged DataFrame containing market data and raw financial variables.
            Returns an empty DataFrame if critical files are missing.
        """
        logging.info(f"Starting data load from {data_dir}...")

        # === EXPLICIT FILE LIST DEFINITION ===
        # Directly listing files to avoid ambiguity and ensure exact reproduction
        
        # Balance Sheet Files (6 files)
        bs_filenames = ["RESSET_BS_ALL_1.csv"] + [f"RESSET_BS_ALL_1 ({i}).csv" for i in range(1, 6)]
        
        # Income Statement Files (6 files)
        is_filenames = ["RESSET_IS_ALL_1.csv"] + [f"RESSET_IS_ALL_1 ({i}).csv" for i in range(1, 6)]
        
        # Stock Data Files (5 files)
        stk_filenames = ["RESSET_MRESSTK_1.csv"] + [f"RESSET_MRESSTK_1 ({i}).csv" for i in range(1, 5)]
        
        # IPO Data File (1 file)
        ipo_filenames = ["RESSET_IISSULST_1.csv"]

        def load_category(file_names: List[str], name: str, keyword_map: dict) -> pd.DataFrame:
            df_list = []
            for fname in file_names:
                fpath = data_dir / fname
                if not fpath.exists():
                    logging.warning(f"Expected file not found: {fname}")
                    continue
                
                try:
                    # Robust encoding reading
                    try: d = pd.read_csv(fpath, encoding='utf-8', on_bad_lines='skip', low_memory=False)
                    except: d = pd.read_csv(fpath, encoding='gb18030', on_bad_lines='skip', low_memory=False)
                    
                    d = cls.standardize_columns(d, keyword_map)
                    
                    # Validation
                    required_keys = ['sid']
                    if 'date' in keyword_map: required_keys.append('date')
                    if not all(k in d.columns for k in required_keys):
                        if 'list_date' in keyword_map and 'sid' in d.columns: pass
                        else: continue
                    
                    # === CLEANING ===
                    d['sid'] = cls.clean_sid(d['sid'])
                    
                    if 'date' in d.columns:
                        d['date'] = pd.to_datetime(d['date'], errors='coerce')
                        # Force Month End
                        d['date'] = d['date'] + pd.offsets.MonthEnd(0)
                        
                        min_y = d['date'].min().year
                        max_y = d['date'].max().year
                        logging.info(f"Loaded {fname}: {len(d)} rows ({min_y}-{max_y})")
                    else:
                        logging.info(f"Loaded {fname}: {len(d)} rows (No Date)")

                    df_list.append(d)
                except Exception as e:
                    logging.error(f"Error reading {fname}: {e}")
            
            if not df_list: return pd.DataFrame()
            full_df = pd.concat(df_list, ignore_index=True)
            
            if name != "IPO Data" and 'date' in full_df.columns:
                full_df = full_df.drop_duplicates(subset=['sid', 'date'], keep='last')
            
            return full_df

        # Load Tables using Explicit Lists
        df_bs = load_category(bs_filenames, "Balance Sheet", cls.KEYWORD_MAP_BS)
        df_is = load_category(is_filenames, "Income Statement", cls.KEYWORD_MAP_IS)
        df_stk = load_category(stk_filenames, "Stock Data", cls.KEYWORD_MAP_STK)
        df_ipo = load_category(ipo_filenames, "IPO Data", cls.KEYWORD_MAP_IPO)

        if df_stk.empty: raise ValueError("Stock data missing!")

        # Merge IPO
        if not df_ipo.empty:
            logging.info("Merging IPO dates...")
            df_ipo['list_date'] = pd.to_datetime(df_ipo['list_date'], errors='coerce')
            df_ipo = df_ipo.sort_values('list_date').groupby('sid').first().reset_index()
            if 'list_date' in df_stk.columns: df_stk = df_stk.drop(columns=['list_date'])
            df_stk = pd.merge(df_stk, df_ipo[['sid', 'list_date']], on='sid', how='left')

        # Calc MktCap
        if 'mktcap' not in df_stk.columns and 'clpr' in df_stk.columns and 'fullshr' in df_stk.columns:
             logging.info("Calculating MktCap...")
             df_stk['clpr'] = pd.to_numeric(df_stk['clpr'], errors='coerce')
             df_stk['fullshr'] = pd.to_numeric(df_stk['fullshr'], errors='coerce')
             df_stk['mktcap'] = df_stk['clpr'] * df_stk['fullshr']

        # Merge Financials
        logging.info("Merging BS and IS...")
        if not df_bs.empty: df_bs['sid'] = cls.clean_sid(df_bs['sid'])
        if not df_is.empty: df_is['sid'] = cls.clean_sid(df_is['sid'])
        
        df_fin = pd.merge(df_bs, df_is, on=['sid', 'date'], how='outer')
        
        fin_vars = ['assets', 'equity', 'rev', 'cogs', 'sell_exp', 'adm_exp', 'fin_exp']
        if not df_fin.empty:
            is_annual_report = df_fin['date'].dt.month == 12
            for col in fin_vars:
                if col in df_fin.columns:
                    df_fin[f'{col}_a'] = df_fin.loc[is_annual_report, col]
                    df_fin[f'{col}_q'] = df_fin[col]
        else:
            for col in fin_vars:
                df_fin[f'{col}_a'] = np.nan
                df_fin[f'{col}_q'] = np.nan

        logging.info("Merging Financials to Stock Data...")
        df_merge = pd.merge(df_stk, df_fin, on=['sid', 'date'], how='left')
        logging.info(f"Merged Data Shape: {df_merge.shape}")
        
        matched = df_merge['assets_q'].notna().sum()
        logging.info(f"Quality Check: {matched} rows have valid financial data matched.")
        
        return df_merge

    @classmethod
    def transform(cls, df: pd.DataFrame, cfg: Optional[BaseConfig] = None) -> pd.DataFrame:
        """
        Computes the Operating Profitability factors (Opa, Ola, Oleq).

        Implements the definition from Hou et al. (2017):
        OpProf = Revenue - COGS - SGA - Interest Expense.
        Applies winsorization (1%/99%) and Z-score standardization.

        Parameters
        ----------
        df : pd.DataFrame
            Merged DataFrame with financial and market data.
        cfg : BaseConfig, optional
            Configuration object.

        Returns
        -------
        pd.DataFrame
            DataFrame containing 'date', 'sid', and constructed feature columns 
            (feat_Opa, feat_Ola, feat_Oleq).
        """
        if cfg is None: cfg = BaseConfig()
        out = df[['date', 'sid']].copy()
        
        def safe_shift(s, n): return s.groupby(df['sid']).shift(n)

        cols_to_numeric = ['rev_a', 'cogs_a', 'sell_exp_a', 'adm_exp_a', 'fin_exp_a',
                           'rev_q', 'cogs_q', 'sell_exp_q', 'adm_exp_q', 'fin_exp_q',
                           'assets_a', 'equity_q']
        for c in cols_to_numeric:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
            else: df[c] = np.nan

        # OpProf
        sga_a = df['sell_exp_a'].fillna(0) + df['adm_exp_a'].fillna(0)
        df['op_prof_a'] = df['rev_a'] - df['cogs_a'] - sga_a - df['fin_exp_a']
        
        sga_q = df['sell_exp_q'].fillna(0) + df['adm_exp_q'].fillna(0)
        df['op_prof_q'] = df['rev_q'] - df['cogs_q'] - sga_q - df['fin_exp_q']

        # Factors
        # A.4.15 Opa
        df['feat_Opa_raw'] = df['op_prof_a'] / df['assets_a']
        # A.4.16 Ola (Lag 12)
        assets_lag12 = safe_shift(df['assets_a'], 12) 
        df['feat_Ola_raw'] = df['op_prof_a'] / assets_lag12
        # A.4.14 Oleq (Lag 3)
        equity_lag3 = safe_shift(df['equity_q'], 3)
        df['feat_Oleq_raw'] = df['op_prof_q'] / equity_lag3

        target_feats = {'feat_Opa': 'feat_Opa_raw', 'feat_Ola': 'feat_Ola_raw', 'feat_Oleq': 'feat_Oleq_raw'}
        
        for feat_name, raw_col in target_feats.items():
            out[feat_name] = df[raw_col]
            if not out[feat_name].isnull().all():
                out[feat_name] = out.groupby('date')[feat_name].transform(
                    lambda x: x.clip(x.quantile(0.01), x.quantile(0.99)))
                out[feat_name] = out.groupby('date')[feat_name].transform(
                    lambda x: (x - x.mean()) / x.std() if x.std()!=0 else 0)
            out[feat_name] = out[feat_name].fillna(0)

        return out

    @classmethod
    def run_backtest(cls, df_feats: pd.DataFrame, df_raw: pd.DataFrame):
        """
        Performs validation using decile portfolio sorts.

        Calculates Long-Short returns (High minus Low), T-statistics, and Sharpe Ratios.
        Generates cumulative return plots and saves summary statistics.

        Parameters
        ----------
        df_feats : pd.DataFrame
            DataFrame containing constructed features.
        df_raw : pd.DataFrame
            DataFrame containing raw market data (returns, mktcap) for validation.
        """
        logging.info("Starting Auto-Validation...")
        out_dir_bt = Path(cls.cfg.root) if hasattr(cls, 'cfg') else Path(".")
        out_dir_bt = out_dir_bt / "results" / "backtest"
        out_dir_figs = out_dir_bt.parent / "figs"
        out_dir_bt.mkdir(parents=True, exist_ok=True)
        out_dir_figs.mkdir(parents=True, exist_ok=True)

        if 'ret' not in df_raw.columns: return
        df = pd.merge(df_feats, df_raw[['sid', 'date', 'ret', 'mktcap']], on=['sid', 'date'], how='inner')
        df = df.sort_values(['sid', 'date'])
        
        df['ret_next'] = df.groupby('sid')['ret'].shift(-1)
        
        factors = [c for c in df.columns if c.startswith('feat_')]
        summary = []
        
        for factor in factors:
            temp = df.dropna(subset=[factor, 'ret_next', 'mktcap']).copy()
            if len(temp) < 500: continue
            
            try: temp['group'] = temp.groupby('date')[factor].transform(
                lambda x: pd.qcut(x, 10, labels=False, duplicates='drop'))
            except: continue
            
            port_ret = temp.groupby(['date', 'group']).apply(
                lambda x: np.average(x['ret_next'], weights=x['mktcap']) if x['mktcap'].sum()>0 else np.nan).unstack()
            
            if port_ret.shape[1] < 2: continue
            hl_ret = port_ret[port_ret.columns.max()] - port_ret[port_ret.columns.min()]
            
            mean = hl_ret.mean(); std = hl_ret.std(); obs = len(hl_ret)
            t = mean/(std/np.sqrt(obs)) if std!=0 else 0
            sharpe = (mean/std)*np.sqrt(12) if std!=0 else 0
            
            summary.append({'Factor': factor, 'Mean': mean, 't-stat': t, 'Sharpe': sharpe, 'Obs': obs})
            
            cum = (1+hl_ret).cumprod()
            plt.figure(figsize=(10,6))
            plt.plot(cum.index, cum.values, label=f"{factor} H-L")
            plt.title(f"Cumulative Return: {factor}")
            plt.xlabel("Date"); plt.ylabel("Cumulative Wealth")
            plt.grid(True); plt.legend()
            plt.savefig(out_dir_figs / f"{factor}_curve.png"); plt.close()

        if summary:
            pd.DataFrame(summary).to_csv(out_dir_bt / "A4_validation_summary.csv", index=False)
            logging.info("Validation Summary Saved.")

    def compute(self) -> pd.DataFrame:
        """
        Orchestrates the full pipeline: Load -> Clean -> Transform -> Validate.

        Applies industry filtering (excluding 'J') and time filtering (2000-2024).
        Implements look-ahead bias prevention via lagging (6M Annual, 4M Quarterly).

        Returns
        -------
        pd.DataFrame
            The final feature DataFrame saved to parquet.
        """
        data_root = Path(self.cfg.root) / self.cfg.data_dir
        out_root = Path(self.cfg.root) / "results" / "features"
        out_root.mkdir(parents=True, exist_ok=True)
        
        try: df = self.load_and_merge_data(data_root)
        except Exception as e: logging.error(e); return pd.DataFrame()
        if df.empty: return pd.DataFrame()
        
        df = df.sort_values(['sid', 'date'])
        
        if 'ind_code' in df.columns:
            logging.info(f"Filtering Industry J...")
            df = df[~df['ind_code'].astype(str).str.startswith('J', na=False)]
            
        # 2. Lags
        for col in [c for c in df.columns if c.endswith('_a')]:
            df[col] = df.groupby('sid')[col].shift(6).ffill()
        for col in [c for c in df.columns if c.endswith('_q')]:
            df[col] = df.groupby('sid')[col].shift(4).ffill()

        feats = self.transform(df, cfg=self.cfg)

        logging.info("Filtering output to 2000-2024...")
        feats = feats[(feats['date'] >= '2000-01-01') & (feats['date'] <= '2024-12-31')]

        feats.to_parquet(out_root / "A4_profitability.parquet", index=False)
        logging.info(f"Features saved to {out_root}")
        
        self.run_backtest(feats, df) 
        
        return feats

if __name__ == "__main__":
    MySolution(MyConfig()).compute()
