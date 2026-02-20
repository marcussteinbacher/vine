from scipy import stats
import numpy as np
from scipy import optimize
from scipy.stats import chi2
import math as m
import pandas as pd
from typing import Union
from numpy.typing import NDArray


def duration_test(violations:pd.Series, conf_level:float=0.95) -> dict:
    """
    Perform the Christoffersen and Pelletier Test (2004) called Duration Test.
    The main objective is to know if the VaR model responds quickly to market movementsin order to do not form volatility clusters.
    Duration is time betwenn violations of VaR.
    This test verifies if violations has no memory i.e. should be independent.
    The duration of time between VaR violations (no-hits) should ideally be independent and not 
    cluster. Under the null hypothesis of a correctly specified risk model, the no-hit duration 
    should have no memory. Since the only continuous distribution which is memory 
    free is the exponential, the test can conducted on any distribution which embeds the exponential 
    as a restricted case, and a likelihood ratio test then conducted to see whether the restriction 
    holds. Following Christoffersen and Pelletier (2004), the Weibull distribution is used with 
    parameter ‘b=1’ representing the case of the exponential.

    Parameters:
        violations (series): boolean series of violations of VaR
        conf_level (float):  test confidence level
    Returns:
        answer (dict):       statistics and decision of the test
    """
    typeok = False
    if isinstance(violations, pd.Series) or isinstance(violations, pd.DataFrame):
        violations = violations.values.flatten()
        typeok = True
    elif isinstance(violations, NDArray):
        violations = violations.flatten()
        typeok = True
    elif isinstance(violations, list):
        typeok = True
    if not typeok:
        raise ValueError("Input must be list, array, series or dataframe.")

    N = int(sum(violations))
    first_hit = violations[0]
    last_hit = violations[-1]

    duration = [i + 1 for i, x in enumerate(violations) if x == 1]

    D = np.diff(duration)

    TN = len(violations)
    C = np.zeros(len(D))

    if not duration or (D.shape[0] == 0 and len(duration) == 0):
        duration = [0]
        D = [0]
        N = 1

    if first_hit == 0:
        C = np.append(1, C)
        D = np.append(duration[0], D)  # days until first violation

    if last_hit == 0:
        C = np.append(C, 1)
        D = np.append(D, TN - duration[-1])

    else:
        N = len(D)

    def likDurationW(x, D, C, N):
        b = x
        a = ((N - C[0] - C[N - 1]) / (sum(D ** b))) ** (1 / b)
        lik = (
            C[0] * np.log(pweibull(D[0], a, b, survival=True))
            + (1 - C[0]) * dweibull(D[0], a, b, log=True)
            + sum(dweibull(D[1 : (N - 1)], a, b, log=True))
            + C[N - 1] * np.log(pweibull(D[N - 1], a, b, survival=True))
            + (1 - C[N - 1]) * dweibull(D[N - 1], a, b, log=True)
        )

        if np.isnan(lik) or np.isinf(lik):
            lik = 1e10
        else:
            lik = -lik
        return lik

    # When b=1 we get the exponential
    def dweibull(D, a, b, log=False):
        # density of Weibull
        pdf = b * np.log(a) + np.log(b) + (b - 1) * np.log(D) - (a * D) ** b
        if not log:
            pdf = np.exp(pdf)
        return pdf

    def pweibull(D, a, b, survival=False):
        # distribution of Weibull
        cdf = 1 - np.exp(-((a * D) ** b))
        if survival:
            cdf = 1 - cdf
        return cdf

    optimizedBetas = optimize.minimize(
        likDurationW, x0=[2], args=(D, C, N), method="L-BFGS-B", bounds=[(0.001, 10)]
    )

    print(optimizedBetas.message)

    b = optimizedBetas.x
    uLL = -likDurationW(b, D, C, N)
    rLL = -likDurationW(np.array([1]), D, C, N)
    LR = 2 * (uLL - rLL)
    LRp = 1 - chi2.cdf(LR, 1)

    H0 = "Duration Between Exceedances have no memory (Weibull b=1 = Exponential)"
    # i.e. whether we fail to reject the alternative in the LR test that b=1 (hence correct model)
    if LRp < (1 - conf_level):
        decision = "Reject H0"
    else:
        decision = "Fail to Reject H0"

    answer = {
        "dur.b": b, #weibull exponential
        "dur.uLL": uLL, #unrestricted log-likelihood value
        "dur.rLL": rLL, # restricted log-likelihood value
        "dur.LRstat": LR, # likelihood ratio
        "dur.LRp": LRp, # likelihood ratio test statistic
        "dur.H0": H0,
        "dur.Decision": decision,
    }

    return answer


def failure_rate(violations:pd.Series) -> dict:
    """
    Simple failure rate.
    Parameters:
        violatios: array of boolean values, with entries 0 (no violation) and 1 (violation).
    """
    if isinstance(violations, pd.Series) or isinstance(violations, pd.DataFrame):
        N = violations.sum()
    elif isinstance(violations, list) or isinstance(violations, NDArray):
        N = sum(violations)
    else:
        raise ValueError("Input must be list, array, series or dataframe.")
    TN = len(violations)

    answer = {"failure rate": N / TN}
    print(f"Failure rate of {round((N/TN)*100,2)}%")
    return answer


def simple_hits(actual:Union[pd.Series,NDArray],var:Union[pd.Series, NDArray])->Union[pd.Series,NDArray]:
    """
    Returns a boolean Series of hits where the actual returns exceed the VaR. 
    On overlapping dates that are not nan.
    """
    if isinstance(actual, pd.Series):
        index = actual.dropna().index.intersection(var.dropna().index)
        violations = actual.loc[index]<var.loc[index]
    else:
        violations = actual<var
    return violations


def kupiec_test(violations:Union[pd.Series,NDArray], conf_level:float=0.95, alpha:float=0.01)->dict:
    """
    Perform Kupiec Test (1995).
    The main goal is to verify if the number of violations, i.e. proportion of failures, is consistent with the
    violations predicted by the model.
       
    Parameters:
        violations (series): series of violations of VaR
        alpha (float): VaR  level
        conf_level (float): test confidence level
    Returns:
        answer (dict): statistics and decision of the test
    """

    n = len(violations)
    I_alpha = sum(violations)
    alpha_hat = I_alpha/n

    LR = 2*m.log(((1-alpha_hat)/(1-alpha))**(n-I_alpha)*(alpha_hat/alpha)**I_alpha)

    critical_chi_square = chi2.ppf(conf_level, 1)  # one degree of freedom

    LRp = 1 - chi2.cdf(LR, 1)

    if LR > critical_chi_square:
        decision = "Reject H0"
    else:
        decision = "Fail to Reject H0"

    return {
        "uc.H0": "Correct Exceedances",
        "uc.LRstat": LR,
        "uc.critical": critical_chi_square,
        "uc.LRp": LRp,
        "uc.Decision": decision,
    }


def christofferson_test(violations:Union[pd.Series,NDArray], conf_level:float=0.95)->dict:
    """
    Implements the conditional (Christoffersen, 1998) coverage tests for the correct number of exceedances.

    Christoffersen, P. (1998), Evaluating Interval Forecasts, International Economic Review, 39, 841–862. \n
    Christoffersen, P., Hahn,J. and Inoue, A. (2001), Testing and Comparing Value-at-Risk Measures, Journal of Empirical Finance, 8, 325–342.
    
    Parameters:
        violations (series): boolean series of violations of VaR, e.g. simple_hits(actual, var)
        conf_level (float): test confidence level
    
    Returns:
        answer (dict): statistics and decision of the test
    """

    df = pd.DataFrame(columns=["T","T+1"])
    if isinstance(violations, pd.Series):
        df.loc[:,"T"] = violations.values[1:]
        df.loc[:,"T+1"] = violations.values[:-1]
    else:
        df.loc[:,"T"] = violations[1:]
        df.loc[:,"T+1"] = violations[:-1]

    def nij(i:bool,j:bool):
        """"
        Returns the number of consecutive days on which j follows i
        """
        return len(df[(df["T"]==i) & (df["T+1"] == j)])

    n = len(violations)
    n00 = nij(False,False)
    n01 = nij(False,True)
    n10 = nij(True,False)
    n11 = nij(True,True)

    pi01 = n01/(n00+n01)
    pi11 = n11/(n10+n11)
    pi = (n01+n11)/(n00+n01+n10+n11)

    LR = -2*m.log(
        ((1-pi)**(n00+n10) * pi**(n01+n11) )
        /
        ((1-pi01)**n00 * pi01**n01 * (1-pi11)**n10 * pi11**n11)
    )

    critical_chi_square = chi2.ppf(conf_level, 1)  # one degree of freedom

    LRp = 1.0 - chi2.cdf(LR, 1)

    if LR > critical_chi_square:
        decision = "Reject H0"
    else:
        decision = "Fail to Reject H0"

    return {
        "ind.H0": "Independent",
        "ind.LRstat": LR,
        "ind.critical": critical_chi_square,
        "ind.LRp": LRp,
        "ind.Decision": decision,
    }


def mcneil_frey_test(actual_returns:Union[pd.Series,NDArray], es_forecasts:Union[pd.Series,NDArray], var_forecasts:Union[pd.Series,NDArray], alpha:float=0.01, conf_level:float=0.95, boot=False, n_boot:int=1000)->dict:
    """
    Übersetzt die R-Implementierung (rugarch/ESTest, Alexios Ghalanos [aut, cre], Tobias Kley [ctb], Initial release 2020-07-14) des Expected Shortfall (ES) Backtests 
    von McNeil und Frey (2000) nach Python.
    Dieser Test prüft, ob das bedingte Expected Shortfall systematisch unterschätzt wird.
    Er führt einen einseitigen Test für die "Excess Shortfalls" durch. Das sind die Beträge,
    um die die tatsächlichen Renditen die ES-Prognose an Tagen übertreffen, an denen ein 
    VaR-Verstoß aufgetreten ist.
    Die Nullhypothese (H0) ist, dass der Mittelwert dieser Überschreitungen null ist.
    Die Alternativhypothese (H1) ist, dass der Mittelwert größer als null ist, was bedeutet,
    dass das ES-Modell das Risiko systematisch unterschätzt.

    Parameter:
    ----------
    actual_returns : array-like
        Eine Serie von tatsächlichen Renditen (Verluste sollten negativ sein).
    es_forecasts : array-like
        Eine Serie von entsprechenden prognostizierten Expected-Shortfall-Werten.
        Diese sollten die gleiche Vorzeichenkonvention wie Renditen haben (d.h. negativ sein).
    var_forecasts : array-like
        Eine Serie von entsprechenden prognostizierten Value-at-Risk-Werten.
        Diese sollten ebenfalls die gleiche Vorzeichenkonvention haben (d.h. negativ sein).
    alpha : float, optional
        Das Signifikanzniveau für den VaR (z.B. 0.01 für 99% VaR). Wird verwendet, um
        die erwartete Anzahl von Überschreitungen zu berechnen. Standard ist 0.01.
    conf_level : float, optional
        Das Konfidenzniveau für den Hypothesentest (z.B. 0.95 für ein
        5% Signifikanzniveau). Standard ist 0.95.
    boot : bool, optional
        Wenn True, wird ein Bootstrap-Test durchgeführt, um einen zusätzlichen p-Wert 
        zu berechnen. Standard ist False.
    n_boot : int, optional
        Die Anzahl der Bootstrap-Wiederholungen, falls boot=True. Standard ist 1000.
    Returns:
    -------
    dict
        Ein Dictionary mit den Ergebnissen des Backtests:
        - 'expected_exceed': Die erwartete Anzahl von VaR-Verstößen.
        - 'actual_exceed': Die tatsächliche Anzahl von VaR-Verstößen.
        - 'H1': Die Aussage der Alternativhypothese.
        - 'boot_p_value': Der gebootstrappte p-Wert (oder NaN, falls boot=False).
        - 'p_value': Der p-Wert aus dem Test.
        - 'Decision': Die Testentscheidung zum gegebenen Konfidenzniveau.
    """

    def _calculate_p_value(data):
        """
        Hilfsfunktion zur Berechnung des p-Wertes für den einseitigen Test. Dies ist eine direkte Übersetzung der .fn-Funktion in R.
        H0: Der Mittelwert der Überschreitungen ist 0.
        H1: Der Mittelwert der Überschreitungen ist größer als 0.
        """
        num_violations = len(data)

        # Der Test erfordert mindestens 2 Verstöße, um die Standardabweichung zu berechnen.
        if num_violations <= 1:
            return np.nan
        mean_excess = np.mean(data)

        # Berechne die Stichproben-Standardabweichung (ddof=1 für n-1 im Nenner, wie bei sd() in R).
        std_dev = np.std(data, ddof=1)
        # Wenn alle Überschreitungswerte identisch sind, ist die Standardabweichung 0. Der Test ist in diesem Fall nicht aussagekräftig.
        if std_dev == 0:
            return np.nan
        
        # Teststatistik (t-Statistik).
        t_stat = mean_excess / (std_dev / np.sqrt(num_violations))

        # Berechne den p-Wert unter Annahme einer Normalverteilung (wie pnorm in R).
        # Dies ist ein einseitiger Test, daher betrachten wir das obere Ende der Verteilung.
        p_value = 1 - stats.norm.cdf(t_stat)

        return p_value

    # Konvertiere die Eingaben in NumPy-Arrays für vektorisierte Operationen
    actual = np.asarray(actual_returns)
    es = np.asarray(es_forecasts)
    var = np.asarray(var_forecasts)

    n = len(actual)

    # Überprüfe, ob alle Eingabeserien die gleiche Länge haben
    if n != len(var) or n != len(es):
        raise ValueError("Die Eingabeserien (actual, VaR, ES) müssen die gleiche Länge haben.")
    
    # Violations
    violations_idx = actual < var

    # Der 'Excess Shortfall' (z) ist die Differenz zwischen der ES-Prognose und der tatsächlichen Rendite an den Tagen, an denen ein Verstoß auftrat.
    z = es[violations_idx] - actual[violations_idx]
    num_violations = len(z)

    p_value = _calculate_p_value(z)

    # Optional zweiter gebootstrappter p-Wert
    boot_p_value = np.nan  # Standardmäßig NaN, ähnlich wie NA in R
    if boot:
        # Führe Bootstrap nur durch, wenn genügend Daten zum Sampeln vorhanden sind
        if num_violations > 1:
            # Erzeuge Bootstrap-Stichproben durch Ziehen mit Zurücklegen aus 'z'
            bootstrap_samples = np.random.choice(
                z, size=(n_boot, num_violations), replace=True
            )
        
            # Wende die p-Wert-Berechnung auf jede Bootstrap-Stichprobe an
            bootstrap_p_values = np.apply_along_axis(
                _calculate_p_value, axis=1, arr=bootstrap_samples
            )
        
            # Der endgültige gebootstrappte p-Wert ist der Mittelwert der einzelnen p-Werte
            boot_p_value = np.nanmean(bootstrap_p_values)

    significance_level = 1 - conf_level

    # Die Entscheidung basiert auf dem nicht gebootstrappten p-Wert
    if np.isnan(p_value):
        decision = f"Nicht genügend Verstöße ({num_violations}), um den Test durchzuführen."
    elif p_value < significance_level:
        # Kleiner p-Wert: Nullhypothese wird verworfen.
        decision = f"H0 auf dem {significance_level:.0%}-Signifikanzniveau verwerfen."
    else:
        # Hoher p-Wert: Nullhypothese kann nicht verworfen werden.
        decision = f"H0 auf dem {significance_level:.0%}-Signifikanzniveau nicht verwerfen."

    results = {
        'expected_exceed': int(np.floor(alpha * n)),
        'actual_exceed': num_violations,
        "H0": "Der Mittelwert der Überschreitungen des VaR ist gleich null",
        'H1': "Der Mittelwert der Überschreitungen des VaR ist größer als null (ES wird systematisch unterschätzt)",
        'boot_p_value': boot_p_value,
        'p_value': p_value,
        'Decision': decision
    }

    return results