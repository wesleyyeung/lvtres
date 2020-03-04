"""
The tableone package is used for creating "Table 1" summary statistics for
research papers.
"""

__author__ = "Tom Pollard <tpollard@mit.edu>, Alistair Johnson, Jesse Raffa"
__version__ = "0.6.6"

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats import multitest
from tabulate import tabulate

import modality

# display deprecation warnings
warnings.simplefilter('always', DeprecationWarning)


class InputError(Exception):
    """
    Exception raised for errors in the input.
    """
    pass


class TableOne(object):
    """

    If you use the tableone package, please cite:

    Pollard TJ, Johnson AEW, Raffa JD, Mark RG (2018). tableone: An open source
    Python package for producing summary statistics for research papers.
    JAMIA Open, Volume 1, Issue 1, 1 July 2018, Pages 26-31.
    https://doi.org/10.1093/jamiaopen/ooy012

    Create an instance of the tableone summary table.

    Parameters
    ----------
    data : pandas DataFrame
        The dataset to be summarised. Rows are observations, columns are
        variables.
    columns : list, optional
        List of columns in the dataset to be included in the final table.
    categorical : list, optional
        List of columns that contain categorical variables.
    groupby : str, optional
        Optional column for stratifying the final table (default: None).
    nonnormal : list, optional
        List of columns that contain non-normal variables (default: None).
    pval : bool, optional
        Display computed P-Values (default: False).
    pval_adjust : str, optional
        Method used to adjust P-Values for multiple testing.
        For a complete list, see documentation for statsmodels multipletests.
        Available methods include ::

        `None` : no correction applied.
        `bonferroni` : one-step correction
        `sidak` : one-step correction
        `holm-sidak` : step down method using Sidak adjustments
        `simes-hochberg` : step-up method (independent)
        `hommel` : closed method based on Simes tests (non-negative)

    missing : bool, optional
        Display a count of null values (default: True).
    ddof : int, optional
        Degrees of freedom for standard deviation calculations (default: 1).
    rename : dict, optional
        Dictionary of alternative names for variables.
        e.g. `rename = {'sex':'gender', 'trt':'treatment'}`
    sort : bool or str, optional
        If `True`, sort the variables alphabetically. If a string
        (e.g. `'P-Value'`), sort by the specified column in ascending order.
        Default (`False`) retains the sequence specified in the `columns`
        argument. Currently the only columns supported are: `'Missing'`,
        `'P-Value'`, `'P-Value (adjusted)'`, and `'Test'`.
    limit : int or dict, optional
        Limit to the top N most frequent categories. If int, apply to all
        categorical variables. If dict, apply to the key (e.g. {'sex': 1}).
    order : dict, optional
        Specify an order for categorical variables. Key is the variable, value
        is a list of values in order.  {e.g. 'sex': ['f', 'm', 'other']}
    remarks : bool, optional
        Add remarks on the appropriateness of the summary measures and the
        statistical tests (default: True).
    label_suffix : bool, optional
        Append summary type (e.g. "mean (SD); median [Q1,Q3], n (%); ") to the
        row label (default: False).
    decimals : int or dict, optional
        Number of decimal places to display. An integer applies the rule to all
        variables (default: 1). A dictionary (e.g. `decimals = {'age': 0)`)
        applies the rule per variable, defaulting to 1 place for unspecified
        variables. For continuous variables, applies to all summary statistics
        (e.g. mean and standard deviation). For categorical variables, applies
        to percentage only.

    Attributes
    ----------
    tableone : dataframe
        Summary of the data (i.e., the "Table 1").
    """

    def __init__(self, data, columns=None, categorical=None, groupby=None,
                 nonnormal=None, pval=False, pval_adjust=None, isnull=None,
                 missing=True, ddof=1, labels=None, rename=None, sort=False,
                 limit=None, order=None, remarks=True, label_suffix=False,
                 decimals=1,reverse_missing=False):

        # labels is now rename
        if labels is not None and rename is not None:
            raise TypeError("TableOne received both labels and rename.")
        elif labels is not None:
            warnings.warn("The labels argument is deprecated; use " +
                          "rename instead.", DeprecationWarning)
            self._alt_labels = labels
        else:
            self._alt_labels = rename

        # isnull is now missing
        if isnull is not None:
            warnings.warn("The isnull argument is deprecated; use " +
                          "missing instead.", DeprecationWarning)
            self._isnull = isnull
        else:
            self._isnull = missing

        # groupby should be a string
        if not groupby:
            groupby = ''
        elif groupby and type(groupby) == list:
            groupby = groupby[0]

        # nonnormal should be a string
        if not nonnormal:
            nonnormal = []
        elif nonnormal and type(nonnormal) == str:
            nonnormal = [nonnormal]

        # if the input dataframe is empty, raise error
        if data.empty:
            raise InputError('The input dataframe is empty.')

        # if columns are not specified, use all columns
        if not columns:
            columns = data.columns.values

        # check that the columns exist in the dataframe
        if not set(columns).issubset(data.columns):
            notfound = list(set(columns) - set(data.columns))
            raise InputError("Columns not found in " +
                             "dataset: {}".format(notfound))

        # check for duplicate columns
        dups = data[columns].columns[data[columns].columns.duplicated()].unique()
        if not dups.empty:
            raise InputError("Input contains duplicate " +
                             "columns: {}".format(dups))

        # if categorical not specified, try to identify categorical
        if not categorical and type(categorical) != list:
            categorical = self._detect_categorical_columns(data[columns])

        # ensure that values to order are strings
        if order:
            for k in order:
                order[k] = ["{}".format(v) for v in order[k]]

        if pval and not groupby:
            raise InputError("If pval=True then groupby must be specified.")

        self._columns = list(columns)
        self._continuous = [c for c in columns if c not in categorical + [groupby]]
        self._categorical = categorical
        self._nonnormal = nonnormal
        self._pval = pval
        self._pval_adjust = pval_adjust
        self._sort = sort
        self._groupby = groupby
        # degrees of freedom for standard deviation
        self._ddof = ddof
        self._limit = limit
        self._order = order
        self._remarks = remarks
        self._label_suffix = label_suffix
        self._decimals = decimals
        
        self._reverse_missing = reverse_missing
        
        if self._reverse_missing:
            self._missing_string = 'Count'
        else:
            self._missing_string = 'Missing'
            
        # output column names that cannot be contained in a groupby
        self._reserved_columns = [self._missing_string, 'P-Value', 'Test',
                                  'P-Value (adjusted)']
        if self._groupby:
            self._groupbylvls = sorted(data.groupby(groupby).groups.keys())
            # check that the group levels do not include reserved words
            for level in self._groupbylvls:
                if level in self._reserved_columns:
                    raise InputError('Group level contains "{}", a reserved' +
                                     ' keyword.'.format(level))
        else:
            self._groupbylvls = ['Overall']

        # forgive me jraffa
        if self._pval:
            self._significance_table = self._create_significance_table(data)

        # correct for multiple testing
        if self._pval and self._pval_adjust:
            alpha = 0.05
            adjusted = multitest.multipletests(self._significance_table['P-Value'],
                                               alpha=alpha,
                                               method=self._pval_adjust)
            self._significance_table['P-Value (adjusted)'] = adjusted[1]
            self._significance_table['adjust method'] = self._pval_adjust

        # create descriptive tables
        if self._categorical:
            self.cat_describe = self._create_cat_describe(data)
            self.cat_table = self._create_cat_table(data)

        # create continuous tables
        if self._continuous:
            self.cont_describe = self._create_cont_describe(data)
            self.cont_table = self._create_cont_table(data)

        # combine continuous variables and categorical variables into table 1
        self.tableone = self._create_tableone(data)
        # self._remarks_str = self._generate_remark_str()

        # wrap dataframe methods
        self.head = self.tableone.head
        self.tail = self.tableone.tail
        self.to_csv = self.tableone.to_csv
        self.to_excel = self.tableone.to_excel
        self.to_html = self.tableone.to_html
        self.to_json = self.tableone.to_json
        self.to_latex = self.tableone.to_latex

    def __str__(self):
        return self.tableone.to_string() + self._generate_remark_str('\n')

    def __repr__(self):
        return self.tableone.to_string() + self._generate_remark_str('\n')

    def _repr_html_(self):
        return self.tableone._repr_html_() + self._generate_remark_str('<br />')

    def tabulate(self, headers=None, tablefmt='grid', **kwargs):
        """
        Pretty-print tableone data. Wrapper for the Python 'tabulate' library.

        Args:
            headers (list): Defines a list of column headers to be used.
            tablefmt (str): Defines how the table is formatted. Table formats
                include: 'plain','simple','github','grid','fancy_grid','pipe',
                'orgtbl','jira','presto','psql','rst','mediawiki','moinmoin',
                'youtrack','html','latex','latex_raw','latex_booktabs',
                and 'textile'.

        Examples:
            To output tableone in github syntax, call tabulate with the
                'tablefmt="github"' argument.

            >>> print(tableone.tabulate(tablefmt='fancy_grid'))
        """
        # reformat table for tabulate
        df = self.tableone

        if not headers:
            try:
                headers = df.columns.levels[1]
            except AttributeError:
                headers = df.columns

        df = df.reset_index()
        df = df.set_index('level_0')
        isdupe = df.index.duplicated()
        df.index = df.index.where(~isdupe, '')
        df = df.rename_axis(None).rename(columns={'level_1': ''})

        return tabulate(df, headers=headers, tablefmt=tablefmt, **kwargs)

    def _generate_remark_str(self, end_of_line='\n'):
        """
        Generate a series of remarks that the user should consider
        when interpreting the summary statistics.
        """
        warnings = {}
        msg = '{}'.format(end_of_line)

        # generate warnings for continuous variables
        if self._continuous:
            # highlight far outliers
            outlier_mask = self.cont_describe.far_outliers > 1
            outlier_vars = list(self.cont_describe.far_outliers[outlier_mask].dropna(how='all').index)
            if outlier_vars:
                warnings["Warning, Tukey test indicates far " +
                         "outliers in"] = outlier_vars

            # highlight possible multimodal distributions using hartigan's dip
            # test -1 values indicate NaN
            modal_mask = (self.cont_describe.diptest >= 0) & (self.cont_describe.diptest <= 0.05)
            modal_vars = list(self.cont_describe.diptest[modal_mask].dropna(how='all').index)
            if modal_vars:
                warnings["Warning, Hartigan's Dip Test reports possible " +
                         "multimodal distributions for"] = modal_vars

            # highlight non normal distributions
            # -1 values indicate NaN
            modal_mask = (self.cont_describe.normaltest >= 0) & (self.cont_describe.normaltest <= 0.001)
            modal_vars = list(self.cont_describe.normaltest[modal_mask].dropna(how='all').index)
            if modal_vars:
                warnings["Warning, test for normality reports " +
                         "non-normal distributions for"] = modal_vars

        # create the warning string
        for n, k in enumerate(sorted(warnings)):
            msg += '[{}] {}: {}.{}'.format(n+1, k, ', '.join(warnings[k]),
                                           end_of_line)

        return msg

    def _detect_categorical_columns(self, data):
        """
        Detect categorical columns if they are not specified.

        Parameters
        ----------
            data : pandas DataFrame
                The input dataset.

        Returns
        ----------
            likely_cat : list
                List of variables that appear to be categorical.
        """
        # assume all non-numerical and date columns are categorical
        numeric_cols = set(data._get_numeric_data().columns.values)
        date_cols = set(data.select_dtypes(include=[np.datetime64]).columns)
        likely_cat = set(data.columns) - numeric_cols
        likely_cat = list(likely_cat - date_cols)
        # check proportion of unique values if numerical
        for var in data._get_numeric_data().columns:
            likely_flag = 1.0 * data[var].nunique()/data[var].count() < 0.05
            if likely_flag:
                likely_cat.append(var)
        return likely_cat

    def _q25(self, x):
        """
        Compute percentile (25th)
        """
        return np.nanpercentile(x.values, 25)

    def _q75(self, x):
        """
        Compute percentile (75th)
        """
        return np.nanpercentile(x.values, 75)

    def _std(self, x):
        """
        Compute standard deviation with ddof degrees of freedom
        """
        return np.nanstd(x.values, ddof=self._ddof)

    def _diptest(self, x):
        """
        Compute Hartigan Dip Test for modality.

        p < 0.05 suggests possible multimodality.
        """
        p = modality.hartigan_diptest(x.values)
        # dropna=False argument in pivot_table does not function as expected
        # https://github.com/pandas-dev/pandas/issues/22159
        # return -1 instead of None
        if pd.isnull(p):
            return -1
        return p

    def _normaltest(self, x):
        """
        Compute test for normal distribution.

        Null hypothesis: x comes from a normal distribution
        p < alpha suggests the null hypothesis can be rejected.
        """
        if len(x.values[~np.isnan(x.values)]) > 10:
            stat, p = stats.normaltest(x.values, nan_policy='omit')
        else:
            p = None
        # dropna=False argument in pivot_table does not function as expected
        # return -1 instead of None
        if pd.isnull(p):
            return -1
        return p

    def _tukey(self, x, threshold):
        """
        Count outliers according to Tukey's rule.

        Where Q1 is the lower quartile and Q3 is the upper quartile,
        an outlier is an observation outside of the range:

        [Q1 - k(Q3 - Q1), Q3 + k(Q3 - Q1)]

        k = 1.5 indicates an outlier
        k = 3.0 indicates an outlier that is "far out"
        """
        vals = x.values[~np.isnan(x.values)]

        try:
            q1, q3 = np.percentile(vals, [25, 75])
            iqr = q3 - q1
            low_bound = q1 - (iqr * threshold)
            high_bound = q3 + (iqr * threshold)
            outliers = np.where((vals > high_bound) | (vals < low_bound))
        except IndexError:
            outliers = []

        return outliers

    def _outliers(self, x):
        """
        Compute number of outliers
        """
        outliers = self._tukey(x, threshold=1.5)
        return np.size(outliers)

    def _far_outliers(self, x):
        """
        Compute number of "far out" outliers
        """
        outliers = self._tukey(x, threshold=3.0)
        return np.size(outliers)

    def _t1_summary(self, x):
        """
        Compute median [IQR] or mean (Std) for the input series.

        Parameters
        ----------
            x : pandas Series
                Series of values to be summarised.
        """
        # set decimal places
        if isinstance(self._decimals, int):
            n = self._decimals
        elif isinstance(self._decimals, dict):
            try:
                n = self._decimals[x.name]
            except KeyError:
                n = 1
        else:
            n = 1
            warnings.warn("The decimals arg must be an int or dict. " +
                          "Defaulting to {} d.p.".format(n))

        if x.name in self._nonnormal:
            f = '{{:.{}f}} [{{:.{}f}},{{:.{}f}}]'.format(n, n, n)
            return f.format(np.nanmedian(x.values),
                            np.nanpercentile(x.values, 25),
                            np.nanpercentile(x.values, 75))
        else:
            f = '{{:.{}f}} ({{:.{}f}})'.format(n, n)
            return f.format(np.nanmean(x.values),
                            np.nanstd(x.values, ddof=self._ddof))

    def _create_cont_describe(self, data):
        """
        Describe the continuous data.

        Parameters
        ----------
            data : pandas DataFrame
                The input dataset.

        Returns
        ----------
            df_cont : pandas DataFrame
                Summarise the continuous variables.
        """
        aggfuncs = [pd.Series.count, np.mean, np.median, self._std,
                    self._q25, self._q75, min, max, self._t1_summary,
                    self._diptest, self._outliers, self._far_outliers,
                    self._normaltest]

        # coerce continuous data to numeric
        cont_data = data[self._continuous].apply(pd.to_numeric,
                                                 errors='coerce')
        # check all data in each continuous column is numeric
        bad_cols = cont_data.count() != data[self._continuous].count()
        bad_cols = cont_data.columns[bad_cols]
        if len(bad_cols) > 0:
            raise InputError("The following continuous column(s) have " +
                             "non-numeric values: {}. Either specify the " +
                             "column(s) as categorical or remove the " +
                             "non-numeric values.""".format(bad_cols.values))

        # check for coerced column containing all NaN to warn user
        for column in cont_data.columns[cont_data.count() == 0]:
            self._non_continuous_warning(column)

        if self._groupby:
            # add the groupby column back
            cont_data = cont_data.merge(data[[self._groupby]],
                                        left_index=True,
                                        right_index=True)

            # group and aggregate data
            df_cont = pd.pivot_table(cont_data,
                                     columns=[self._groupby],
                                     aggfunc=aggfuncs)
        else:
            # if no groupby, just add single group column
            df_cont = cont_data.apply(aggfuncs).T
            df_cont.columns.name = 'Overall'
            df_cont.columns = pd.MultiIndex.from_product([df_cont.columns,
                                                         ['Overall']])

        df_cont.index = df_cont.index.rename('variable')

        # remove prefix underscore from column names (e.g. _std -> std)
        agg_rename = df_cont.columns.levels[0]
        agg_rename = [x[1:] if x[0] == '_' else x for x in agg_rename]
        df_cont.columns = df_cont.columns.set_levels(agg_rename, level=0)

        return df_cont

    def _format_cat(self, row):
        var = row.name[0]
        if var in self._decimals:
            n = self._decimals[var]
        else:
            n = 1
        f = '{{:.{}f}}'.format(n)
        return f.format(row.percent)

    def _create_cat_describe(self, data):
        """
        Describe the categorical data.

        Parameters
        ----------
            data : pandas DataFrame
                The input dataset.

        Returns
        ----------
            df_cat : pandas DataFrame
                Summarise the categorical variables.
        """
        group_dict = {}

        for g in self._groupbylvls:
            if self._groupby:
                d_slice = data.loc[data[self._groupby] == g, self._categorical]
            else:
                d_slice = data[self._categorical].copy()

            # create a dataframe with freq, proportion
            df = d_slice.copy()

            # convert to str to handle int converted to boolean. Avoid nans.
            for column in df.columns:
                df[column] = [str(row) if not pd.isnull(row)
                              else None for row in df[column].values]

            df = df.melt().groupby(['variable',
                                    'value']).size().to_frame(name='freq')

            df['percent'] = df['freq'].div(df.freq.sum(level=0),
                                           level=0).astype(float) * 100

            # set number of decimal places for percent
            if isinstance(self._decimals, int):
                n = self._decimals
                f = '{{:.{}f}}'.format(n)
                df['percent'] = df['percent'].astype(float).map(f.format)
            elif isinstance(self._decimals, dict):
                df.loc[:, 'percent'] = df.apply(self._format_cat, axis=1)
            else:
                n = 1
                f = '{{:.{}f}}'.format(n)
                df['percent'] = df['percent'].astype(float).map(f.format)

            # add n column, listing total non-null values for each variable
            ct = d_slice.count().to_frame(name='n')
            ct.index.name = 'variable'
            df = df.join(ct)

            # add null count
            if self._reverse_missing:
                nulls = len(d_slice) - d_slice.isnull().sum().to_frame(name=self._missing_string)
            else:
                nulls = d_slice.isnull().sum().to_frame(name=self._missing_string)
            nulls.index.name = 'variable'
            # only save null count to the first category for each variable
            # do this by extracting the first category from the df row index
            levels = df.reset_index()[['variable',
                                       'value']].groupby('variable').first()
            # add this category to the nulls table
            nulls = nulls.join(levels)
            nulls = nulls.set_index('value', append=True)
            # join nulls to categorical
            df = df.join(nulls)

            # add summary column
            df['t1_summary'] = df.freq.map(str) + ' (' + df.percent.map(str) + ')'

            # add to dictionary
            group_dict[g] = df

        df_cat = pd.concat(group_dict, axis=1)
        # ensure the groups are the 2nd level of the column index
        if df_cat.columns.nlevels > 1:
            df_cat = df_cat.swaplevel(0, 1, axis=1).sort_index(axis=1, level=0)

        return df_cat

    def _create_significance_table(self, data):
        """
        Create a table containing P-Values for significance tests. Add features
        of the distributions and the P-Values to the dataframe.

        Parameters
        ----------
            data : pandas DataFrame
                The input dataset.

        Returns
        ----------
            df : pandas DataFrame
                A table containing the P-Values, test name, etc.
        """
        # list features of the variable e.g. matched, paired, n_expected
        df = pd.DataFrame(index=self._continuous+self._categorical,
                          columns=['continuous', 'nonnormal',
                                   'min_observed', 'P-Value', 'Test'])

        df.index = df.index.rename('variable')
        df['continuous'] = np.where(df.index.isin(self._continuous),
                                    True, False)

        df['nonnormal'] = np.where(df.index.isin(self._nonnormal),
                                   True, False)

        # list values for each variable, grouped by groupby levels
        for v in df.index:
            is_continuous = df.loc[v]['continuous']
            is_categorical = ~df.loc[v]['continuous']
            is_normal = ~df.loc[v]['nonnormal']

            # if continuous, group data into list of lists
            if is_continuous:
                catlevels = None
                grouped_data = []
                for s in self._groupbylvls:
                    lvl_data = data.loc[data[self._groupby] == s, v]
                    # coerce to numeric and drop non-numeric data
                    lvl_data = lvl_data.apply(pd.to_numeric,
                                              errors='coerce').dropna()
                    # append to overall group data
                    grouped_data.append(lvl_data.values)
                min_observed = len(min(grouped_data, key=len))
            # if categorical, create contingency table
            elif is_categorical:
                catlevels = sorted(data[v].astype('category').cat.categories)
                grouped_data = pd.crosstab(data[self._groupby].rename('_groupby_var_'),
                                           data[v])
                min_observed = grouped_data.sum(axis=1).min()

            # minimum number of observations across all levels
            df.loc[v, 'min_observed'] = min_observed

            # compute pvalues
            df.loc[v, 'P-Value'], df.loc[v, 'Test'] = self._p_test(v,
                                                                   grouped_data,
                                                                   is_continuous,
                                                                   is_categorical,
                                                                   is_normal,
                                                                   min_observed,
                                                                   catlevels)

        return df

    def _p_test(self, v, grouped_data, is_continuous, is_categorical,
                is_normal, min_observed, catlevels):
        """
        Compute P-Values.

        Parameters
        ----------
            v : str
                Name of the variable to be tested.
            grouped_data : list
                List of lists of values to be tested.
            is_continuous : bool
                True if the variable is continuous.
            is_categorical : bool
                True if the variable is categorical.
            is_normal : bool
                True if the variable is normally distributed.
            min_observed : int
                Minimum number of values across groups for the variable.
            catlevels : list
                Sorted list of levels for categorical variables.

        Returns
        ----------
            pval : float
                The computed P-Value.
            ptest : str
                The name of the test used to compute the P-Value.
        """

        # no test by default
        pval = np.nan
        ptest = 'Not tested'

        # do not test if the variable has no observations in a level
        if min_observed == 0:
            warnings.warn("No P-Value was computed for {} due to the low " +
                          "number of observations.".format(v))
            return pval, ptest

        # continuous
        if is_continuous and is_normal and len(grouped_data) == 2:
            ptest = 'Two Sample T-test'
            test_stat, pval = stats.ttest_ind(*grouped_data, equal_var=False)
        elif is_continuous and is_normal:
            # normally distributed
            ptest = 'One-way ANOVA'
            test_stat, pval = stats.f_oneway(*grouped_data)
        elif is_continuous and not is_normal:
            # non-normally distributed
            ptest = 'Kruskal-Wallis'
            test_stat, pval = stats.kruskal(*grouped_data)
        # categorical
        elif is_categorical:
            # default to chi-squared
            ptest = 'Chi-squared'
            chi2, pval, dof, expected = stats.chi2_contingency(grouped_data)
            # if any expected cell counts are < 5, chi2 may not be valid
            # if this is a 2x2, switch to fisher exact
            if expected.min() < 5:
                if grouped_data.shape == (2, 2):
                    ptest = "Fisher's exact"
                    oddsratio, pval = stats.fisher_exact(grouped_data)
                else:
                    ptest = 'Chi-squared (warning: expected count < 5)'
                    warnings.warn("Chi-squared test for {} may be invalid " +
                                  "(expected cell counts are < 5).".format(v))

        return pval, ptest

    def _create_cont_table(self, data):
        """
        Create tableone for continuous data.

        Returns
        ----------
        table : pandas DataFrame
            A table summarising the continuous variables.
        """
        # remove the t1_summary level
        table = self.cont_describe[['t1_summary']].copy()
        table.columns = table.columns.droplevel(level=0)

        # add a column of null counts as 1-count() from previous function
        if self._reverse_missing:
            nulltable = len(data) - data[self._continuous].isnull().sum().to_frame(name=self._missing_string)
        else:
            nulltable = data[self._continuous].isnull().sum().to_frame(name=self._missing_string)
        try:
            table = table.join(nulltable)
        # if columns form a CategoricalIndex, need to convert to string first
        except TypeError:
            table.columns = table.columns.astype(str)
            table = table.join(nulltable)

        # add an empty value column, for joining with cat table
        table['value'] = ''
        table = table.set_index([table.index, 'value'])

        # add pval column
        if self._pval and self._pval_adjust:
            table = table.join(self._significance_table[['P-Value (adjusted)',
                                                        'Test']])
        elif self._pval:
            table = table.join(self._significance_table[['P-Value', 'Test']])

        return table

    def _create_cat_table(self, data):
        """
        Create table one for categorical data.

        Returns
        ----------
        table : pandas DataFrame
            A table summarising the categorical variables.
        """
        table = self.cat_describe['t1_summary'].copy()
        # add the total count of null values across all levels
        if self._reverse_missing:
            isnull = len(data) - data[self._categorical].isnull().sum().to_frame(name=self._missing_string)
        else:
            isnull = data[self._categorical].isnull().sum().to_frame(name=self._missing_string)
        isnull.index = isnull.index.rename('variable')
        try:
            table = table.join(isnull)
        # if columns form a CategoricalIndex, need to convert to string first
        except TypeError:
            table.columns = table.columns.astype(str)
            table = table.join(isnull)

        # add pval column
        if self._pval and self._pval_adjust:
            table = table.join(self._significance_table[['P-Value (adjusted)',
                                                         'Test']])
        elif self._pval:
            table = table.join(self._significance_table[['P-Value', 'Test']])

        return table

    def _create_tableone(self, data):
        """
        Create table 1 by combining the continuous and categorical tables.

        Returns
        ----------
        table : pandas DataFrame
            The complete table one.
        """
        if self._continuous and self._categorical:

            # support pandas<=0.22
            try:
                table = pd.concat([self.cont_table, self.cat_table],
                                  sort=False)
            except TypeError:
                table = pd.concat([self.cont_table, self.cat_table])
        elif self._continuous:
            table = self.cont_table
        elif self._categorical:
            table = self.cat_table

        # ensure column headers are strings before reindexing
        table = table.reset_index().set_index(['variable', 'value'])
        table.columns = table.columns.values.astype(str)

        # sort the table rows
        sort_columns = [self._missing_string, 'P-Value', 'P-Value (adjusted)', 'Test']
        if self._sort and isinstance(self._sort, bool):
            new_index = sorted(table.index.values, key=lambda x: x[0].lower())
        elif self._sort and isinstance(self._sort, str) and (self._sort in
                                                             sort_columns):
            try:
                new_index = table.sort_values(self._sort).index
            except KeyError:
                new_index = sorted(table.index.values,
                                   key=lambda x: self._columns.index(x[0]))
                warnings.warn('Sort variable not found: {}'.format(self._sort))
        elif self._sort and isinstance(self._sort, str) and (self._sort not in
                                                             sort_columns):
            new_index = sorted(table.index.values,
                               key=lambda x: self._columns.index(x[0]))
            warnings.warn('Sort must be in the following ' +
                          'list: {}.'.format(self._sort))
        else:
            # sort by the columns argument
            new_index = sorted(table.index.values,
                               key=lambda x: self._columns.index(x[0]))
        table = table.reindex(new_index)

        # round pval column and convert to string
        if self._pval and self._pval_adjust:
            table['P-Value (adjusted)'] = table['P-Value (adjusted)'].apply('{:.3f}'.format).astype(str)
            table.loc[table['P-Value (adjusted)'] == '0.000',
                      'P-Value (adjusted)'] = '<0.001'
        elif self._pval:
            table['P-Value'] = table['P-Value'].apply('{:.3f}'.format).astype(str)
            table.loc[table['P-Value'] == '0.000', 'P-Value'] = '<0.001'

        # if an order is specified, apply it
        if self._order:
            for k in self._order:

                # Skip if the variable isn't present
                try:
                    all_var = table.loc[k].index.unique(level='value')
                except KeyError:
                    warnings.warn('Order variable not found: {}'.format(k))
                    continue

                # Remove value from order if it is not present
                if [i for i in self._order[k] if i not in all_var]:
                    rm_var = [i for i in self._order[k] if i not in all_var]
                    self._order[k] = [i for i in self._order[k]
                                      if i in all_var]
                    warnings.warn('Order value not found: {}: {}'.format(k,
                                                                         rm_var))

                new_seq = [(k, '{}'.format(v)) for v in self._order[k]]
                new_seq += [(k, '{}'.format(v)) for v in all_var
                            if v not in self._order[k]]

                # restructure to match the original idx
                new_idx_array = np.empty((len(new_seq),), dtype=object)
                new_idx_array[:] = [tuple(i) for i in new_seq]
                orig_idx = table.index.values.copy()
                orig_idx[table.index.get_loc(k)] = new_idx_array
                table = table.reindex(orig_idx)

        # set the limit on the number of categorical variables
        if self._limit:
            levelcounts = data[self._categorical].nunique()
            for k, _ in levelcounts.iteritems():

                # set the limit for the variable
                if (isinstance(self._limit, int)
                        and levelcounts[k] >= self._limit):
                    limit = self._limit
                elif isinstance(self._limit, dict) and k in self._limit:
                    limit = self._limit[k]
                else:
                    continue

                if not self._order or (self._order and k not in self._order):
                    # re-order the variables by frequency
                    count = data[k].value_counts().sort_values(ascending=False)
                    new_idx = [(k, '{}'.format(i)) for i in count.index]
                else:
                    # apply order
                    all_var = table.loc[k].index.unique(level='value')
                    new_idx = [(k, '{}'.format(v)) for v in self._order[k]]
                    new_idx += [(k, '{}'.format(v)) for v in all_var
                                if v not in self._order[k]]

                # restructure to match the original idx
                new_idx_array = np.empty((len(new_idx),), dtype=object)
                new_idx_array[:] = [tuple(i) for i in new_idx]
                orig_idx = table.index.values.copy()
                orig_idx[table.index.get_loc(k)] = new_idx_array
                table = table.reindex(orig_idx)

                # drop the rows > the limit
                table = table.drop(new_idx_array[limit:])

        # insert n row
        n_row = pd.DataFrame(columns=['variable', 'value', self._missing_string])
        n_row = n_row.set_index(['variable', 'value'])
        n_row.loc['n', self._missing_string] = None
        
        # support pandas<=0.22
        try:
            table = pd.concat([n_row, table], sort=False)
        except TypeError:
            table = pd.concat([n_row, table])

        if self._groupbylvls == ['Overall']:
            table.loc['n', 'Overall'] = len(data.index)
        else:
            for g in self._groupbylvls:
                ct = data[self._groupby][data[self._groupby] == g].count()
                table.loc['n', '{}'.format(g)] = ct

        # only display data in first level row
        dupe_mask = table.groupby(level=[0]).cumcount().ne(0)
        dupe_columns = [self._missing_string]
        optional_columns = ['P-Value', 'P-Value (adjusted)', 'Test']
        for col in optional_columns:
            if col in table.columns.values:
                dupe_columns.append(col)

        table[dupe_columns] = table[dupe_columns].mask(dupe_mask).fillna('')

        # remove Missing column if not needed
        if not self._isnull:
            table = table.drop(self._missing_string, axis=1) 

        # replace nans with empty strings
        table = table.fillna('')

        # add column index
        if not self._groupbylvls == ['Overall']:
            # rename groupby variable if requested
            c = self._groupby
            if self._alt_labels:
                if self._groupby in self._alt_labels:
                    c = self._alt_labels[self._groupby]

            c = 'Grouped by {}'.format(c)
            table.columns = pd.MultiIndex.from_product([[c], table.columns])

        # display alternative labels if assigned
        table = table.rename(index=self._create_row_labels(), level=0)

        # ensure the order of columns is consistent
        if self._groupby and self._order and (self._groupby in self._order):
            header = ['{}'.format(v) for v in table.columns.levels[1].values]
            cols = self._order[self._groupby] + ['{}'.format(v)
                                                 for v in header
                                                 if v not in
                                                 self._order[self._groupby]]
        elif self._groupby:
            cols = ['{}'.format(v) for v in table.columns.levels[1].values]
        else:
            cols = ['{}'.format(v) for v in table.columns.values]

        cols = [self._missing_string] + [x for x in cols if x != self._missing_string]

        # move optional_columns to the end of the dataframe
        for col in optional_columns:
            if col in cols:
                cols = [x for x in cols if x != col] + [col]

        if self._groupby:
            table = table.reindex(cols, axis=1, level=1)
        else:
            table = table.reindex(cols, axis=1)

        try:
            if self._missing_string in self._alt_labels or 'Overall' in self._alt_labels:
                table = table.rename(columns=self._alt_labels)
        except TypeError:
            pass

        # remove the 'variable, value' column names in the index
        table = table.rename_axis([None, None])

        return table

    def _create_row_labels(self):
        """
        Take the original labels for rows. Rename if alternative labels are
        provided. Append label suffix if label_suffix is True.

        Returns
        ----------
        labels : dictionary
            Dictionary, keys are original column name, values are final label.

        """
        # start with the original column names
        labels = {}
        for c in self._columns:
            labels[c] = c

        # replace column names with alternative names if provided
        if self._alt_labels:
            for k in self._alt_labels.keys():
                labels[k] = self._alt_labels[k]

        # append the label suffix
        if self._label_suffix:
            for k in labels.keys():
                if k in self._nonnormal:
                    labels[k] = "{}, {}".format(labels[k], "median [Q1,Q3]")
                elif k in self._categorical:
                    labels[k] = "{}, {}".format(labels[k], "n (%)")
                else:
                    labels[k] = "{}, {}".format(labels[k], "mean (SD)")

        return labels

    # warnings
    def _non_continuous_warning(self, c):
        warnings.warn('"{}" has all non-numeric values. Consider including ' +
                      'it in the list of categorical ' +
                      'variables.'.format(c), RuntimeWarning, stacklevel=2)
