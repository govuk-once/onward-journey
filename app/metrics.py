import pandas as pd

def clarification_success_gain_metric(df_clarity: pd.DataFrame, df_forced: pd.DataFrame) -> dict:
    
    """Calculates the Clarification Success Gain (CSG)."""
    
    df_clarity_amb = df_clarity[df_clarity['is_ambiguous'] == True].copy()
    df_forced_amb  = df_forced[df_forced['is_ambiguous'] == True].copy()
    
    if df_clarity_amb.empty or df_forced_amb.empty: return {}
         
    df_clarity_amb = df_clarity_amb.set_index('test_id')
    df_forced_amb  = df_forced_amb.set_index('test_id')
    
    ts_clarity = (df_clarity_amb['match_status'] == 'PASS').mean()
    ts_initial = (df_forced_amb['match_status'] == 'PASS').mean()
    
    COST_PENALTY_C = 0.0 # Placeholder for cost penalty
    csg            = (ts_clarity - ts_initial) - COST_PENALTY_C
    
    metrics = {
        'total_ambiguous_cases': len(df_clarity_amb),
        'ts_clarity_accuracy': round(ts_clarity, 4),
        'ts_initial_accuracy': round(ts_initial, 4),
        'clarification_success_gain_csg': round(csg, 4),
    }
    return metrics