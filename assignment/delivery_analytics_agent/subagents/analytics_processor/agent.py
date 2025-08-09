from google.adk.agents import Agent
import pandas as pd
import numpy as np
import google.generativeai as genai
from typing import Dict, List, Any, Optional
import os
import json
import re

_delivery_df = None
_payout_df = None

# Tool 1: Load and prepare data with intelligent schema detection
def load_and_prepare_data(csv_path: str) -> Dict[str, Any]:
    """
    Load both delivery data and payout rules, return comprehensive data context
    
    Example:
    Input: csv_path="/path/to/delivery_data.csv"
    Output: {
        'delivery_data': {'dataframe': DataFrame, 'columns': [...], 'shape': (4, 12)},
        'payout_rules': {'dataframe': DataFrame, 'rules_summary': [...]},
        'status': 'success'
    }
    
    Args:
        csv_path (str): Path to user's delivery CSV file
        
    Returns:
        Dict containing both dataframes and comprehensive metadata for analysis
    """
    global _delivery_df, _payout_df
    
    try:

        _delivery_df = pd.read_csv(csv_path)

        payout_csv_path = "/Users/tapasdas/PycharmProjects/google-adk/assignment/delivery_analytics_agent/payout_information.csv"
        _payout_df = pd.read_csv(payout_csv_path)

        data_context = {
            'delivery_data': {
                'preview': _delivery_df.head(3).to_dict('records'),
                'columns': list(_delivery_df.columns),
                'shape': _delivery_df.shape,
                'dtypes': {k: str(v) for k, v in _delivery_df.dtypes.to_dict().items()},
                'sample_data': _delivery_df.head(3).to_dict('records'),
                'numeric_columns': _delivery_df.select_dtypes(include=[np.number]).columns.tolist(),
                'categorical_columns': _delivery_df.select_dtypes(include=['object']).columns.tolist(),
                'unique_values': {
                    col: _delivery_df[col].unique().astype(str).tolist()[:10]
                    for col in _delivery_df.columns
                },
                'summary_stats': _delivery_df.describe().to_dict()
            },
            'payout_rules': {
                'preview': _payout_df.head(3).to_dict('records'),
                'columns': list(_payout_df.columns),
                'rules_summary': _payout_df.to_dict('records'),
                'categories': (
                    _payout_df['Category'].unique().tolist()
                    if 'Category' in _payout_df.columns
                    else []
                )
            },
            'combined_context': {
                'total_delivery_partners': len(_delivery_df),
                'zones_available': (
                    _delivery_df['Delivery Zone'].unique().tolist()
                    if 'Delivery Zone' in _delivery_df.columns
                    else []
                ),
                'date_range': 'Current data snapshot'
            },
            'status': 'success'
        }

        return data_context
        
    except Exception as e:
        return {
            'delivery_data': None,
            'payout_rules': None,
            'status': 'ERROR',
            'error_message': str(e)
        }

_cached_judge_model = None

# Tool 2: LLM Judge for pandas code validation (feedback only) with prompt caching
def judge_pandas_code(user_query: str, pandas_code: str, data_context_json: str) -> Dict[str, Any]:
    """
    Judge the generated pandas code and provide feedback using cached schemas for efficiency
    
    Example:
    Input: user_query="Calculate total payouts", pandas_code="result = delivery_df.sum()", data_context_json="{...}"
    Output: {
        'validation_passed': False,
        'feedback': 'VERDICT: NEEDS_IMPROVEMENT\nISSUES: Code does not calculate payouts correctly...',
        'status': 'success'
    }
    
    Args:
        user_query (str): Original natural language query from user
        pandas_code (str): Generated pandas code to be judged
        data_context_json (str): JSON string of complete data context
        
    Returns:
        Dict containing validation result and detailed feedback
    """
    global _cached_judge_model
    
    try:

        data_context = json.loads(data_context_json)

        SYSTEM_CONTEXT = """
        You are an expert code reviewer for delivery analytics. Your task is to evaluate pandas code against user queries and data schemas.
        
        DELIVERY DATA SCHEMA (delivery boys database context):
        - Name: The full name of the delivery person.
        - Number of Deliveries: The total count of successful deliveries completed by the person.
        - Distance travelled (km): The total distance covered by the delivery person in kilometers.
        - Delivery Zone: The primary geographical zone assigned to or served by the delivery person (e.g., "Zone A", "Zone B").
        - Full Shift (hours): The number of hours worked indicating completion of a full shift.
        - Customer Feedback (Rating 1-5): The average rating received from customers, on a scale of 1 to 5.
        - Peak Hours Deliveries: The number of deliveries completed specifically during designated peak hours.
        - Undelivered/Returns: The count of items that were either undelivered or returned by the customer.
        - Special Package Type: The type of special package handled, categorizing the nature of items delivered (e.g., "Fragile", "Standard", "Documents").
        - Damage/Loss Items: The number of items that were reported as damaged or lost during delivery.
        - Late Reporting (mins): The duration in minutes by which the delivery person reported late.
        - Advance/Salary Reimbursement (INR): Any amount of salary advance or reimbursement taken by the delivery person, in Indian Rupees.
        - Fixed Salary (INR): The predetermined fixed base salary for the delivery person, in Indian Rupees.
        
        PAYOUT RULES SCHEMA:
        Category,Condition,Amount (₹),Notes
        Delivery Payout,Zone A,₹25 per delivery,-
        Delivery Payout,Zone B,₹20 per delivery,-
        Delivery Payout,Zone C,₹15 per delivery,-
        Delivery Payout,Zone D,₹15 per delivery,-
        Kilometer Payout,Per Kilometer,₹3 per km,Applies on total km travelled
        Daily Bonus,Full Shift Present,₹100 per shift,Must have full attendance for the shift
        Peak Hour Bonus,Delivery during Peak Hours,₹5 per delivery,Additional to regular delivery payout
        Undelivered Item,If item undelivered,₹0,No payout for undelivered items
        Damage or Loss,Per Damaged/Lost Item,-₹100,Deduction per Incident
        Late Reporting,If reported late,-₹100,Fixed deduction per late report
        Salary Advance,If salary advance taken,(Amount) deducted,Deduct salary advance from total payout
        
        EVALUATION CRITERIA:
        1. Does the code correctly address the user's query intent?
        2. Does it use the right columns based on available schemas?
        3. Are payout calculations accurate according to the payout rules?
        4. Does it handle data types correctly (especially Amount parsing)?
        5. Is the logic sound for business requirements?
        6. Does it store results in 'result' variable?
        7. Is the code executable and robust?
        
        RESPONSE FORMAT:
        If the code is GOOD and addresses the query correctly:
        VERDICT: APPROVED
        REASON: Brief explanation of why it's correct
        
        If the code needs improvement:
        VERDICT: NEEDS_IMPROVEMENT
        ISSUES: List specific issues found with detailed feedback for improvement
        """

        evaluation_prompt = f"""
        USER QUERY: "{user_query}"
        
        AVAILABLE COLUMNS IN DELIVERY DATA: {data_context['delivery_data']['columns']}
        AVAILABLE COLUMNS IN PAYOUT DATA: {data_context['payout_rules']['columns']}
        
        CURRENT PANDAS CODE TO JUDGE:
        ```python
        {pandas_code}
        ```
        
        Provide your evaluation based on the criteria and format specified:
        """
 
        if _cached_judge_model is None:

            _cached_judge_model = genai.GenerativeModel(
                'gemini-2.0-flash',
                system_instruction=SYSTEM_CONTEXT
            )
            _cached_judge_model = _cached_judge_model.start_chat()
            print("Judge model cached successfully")

        judge_response = _cached_judge_model.send_message(
            evaluation_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=1000,
            )
        )
        
        judge_feedback = judge_response.text.strip()

        validation_passed = "VERDICT: APPROVED" in judge_feedback
        
        return {
            'validation_passed': validation_passed,
            'feedback': judge_feedback,
            'status': 'success'
        }
        
    except Exception as e:
        return {
            'validation_passed': False,
            'feedback': None,
            'status': 'ERROR',
            'error_message': str(e)
        }

# Tool 3: Query understanding and pandas code generation with iterative improvement using judge feedback
def generate_pandas_code(user_query: str, data_context_json: str) -> Dict[str, Any]:
    """
    Generate pandas code with iterative improvement based on judge feedback
    
    Example:
    Input: user_query="Calculate total payouts for each delivery partner", data_context_json="{...}"
    Output: {
        'pandas_code': 'delivery_df_enhanced = delivery_df.copy()\\n# Calculate payouts...',
        'explanation': 'Generated pandas code using Gemini to analyze: Calculate total payouts [Validated through 2 iterations]',
        'judge_feedback': 'VERDICT: APPROVED\\nREASON: Code correctly calculates payouts...',
        'validation_passed': True,
        'status': 'success'
    }
    
    Args:
        user_query (str): Natural language query from user
        data_context_json (str): JSON string of complete data context from load_and_prepare_data
        
    Returns:
        Dict containing validated pandas code, explanation, and judge feedback
    """
    try:
        data_context = json.loads(data_context_json)
        
        delivery_columns = data_context['delivery_data']['columns']
        payout_columns = data_context['payout_rules']['columns']
        sample_data = data_context['delivery_data']['sample_data']
        payout_rules = data_context['payout_rules']['rules_summary']
        numeric_columns = data_context['delivery_data']['numeric_columns']
        categorical_columns = data_context['delivery_data']['categorical_columns']

        model = genai.GenerativeModel('gemini-2.0-flash')
        
        max_iterations = 3
        current_code = None
        judge_feedback_history = []
        
        for iteration in range(max_iterations):
  
            system_prompt = """You are an expert pandas code generator for delivery analytics. 
            Generate ONLY clean, executable pandas code that directly answers the user's query.
            
            IMPORTANT RULES:
            1. Return ONLY the pandas code, no explanations or markdown
            2. Use ONLY 'delivery_df' and 'payout_df' as dataframe names
            3. Store final results in a variable called 'result'
            4. Handle potential errors gracefully
            5. Include necessary imports within the code if needed
            6. For payout calculations, extract numeric values from Amount column using regex or string manipulation
            7. Make code robust and production-ready
            """
            
            context_prompt = f"""
            DATA STRUCTURE:
            
            DELIVERY DATA (delivery_df):
            - Columns: {delivery_columns}
            - Numeric columns: {numeric_columns}
            - Categorical columns: {categorical_columns}
            - Sample row: {sample_data[0] if sample_data else {}}
            - Shape: {data_context['delivery_data']['shape']}
            
            PAYOUT RULES DATA (payout_df):
            - Columns: {payout_columns}
            - Rules available: {payout_rules}
            - Contains rates for zones, bonuses, penalties
            
            USER QUERY: "{user_query}"
            """

            feedback_prompt = ""
            if judge_feedback_history:
                latest_feedback = judge_feedback_history[-1]
                feedback_prompt = f"""
                
                PREVIOUS ITERATION FEEDBACK:
                {latest_feedback}
                
                Please address the issues mentioned in the feedback and generate improved code.
                """
            
            final_prompt = f"""
            {context_prompt}
            {feedback_prompt}
            
            Generate pandas code that:
            1. Directly answers the user's specific question
            2. Uses appropriate aggregations, filters, or calculations
            3. Handles payout calculations by parsing Amount column if needed
            4. Returns meaningful results stored in 'result' variable
            5. Is production-ready and handles edge cases
            {f"6. Addresses the feedback: {latest_feedback}" if judge_feedback_history else ""}
            
            Generate ONLY the executable pandas code:
            """

            response = model.generate_content(
                system_prompt + "\n\n" + final_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=1500,
                )
            )
            
            generated_code = response.text.strip()

            if "```python" in generated_code:
                generated_code = generated_code.split("```python")[1].split("```")[0].strip()
            elif "```" in generated_code:
                generated_code = generated_code.split("```")[1].split("```")[0].strip()
            
            current_code = generated_code

            judge_result = judge_pandas_code(user_query, current_code, data_context_json)
            
            if judge_result['status'] != 'success':

                return {
                    'pandas_code': current_code,
                    'explanation': f"Generated pandas code using Gemini to analyze: {user_query} [Judge validation failed]",
                    'judge_feedback': f"Judge error: {judge_result.get('error_message', 'Unknown error')}",
                    'validation_passed': False,
                    'iterations_used': iteration + 1,
                    'status': 'success'
                }
            
            judge_feedback_history.append(judge_result['feedback'])

            if judge_result['validation_passed']:
                return {
                    'pandas_code': current_code,
                    'explanation': f"Generated pandas code using Gemini to analyze: {user_query} [Validated through {iteration + 1} iterations]",
                    'judge_feedback': judge_result['feedback'],
                    'validation_passed': True,
                    'iterations_used': iteration + 1,
                    'status': 'success'
                }

        return {
            'pandas_code': current_code,
            'explanation': f"Generated pandas code using Gemini to analyze: {user_query} [Max {max_iterations} iterations reached]",
            'judge_feedback': judge_feedback_history[-1] if judge_feedback_history else "No feedback available",
            'validation_passed': False,
            'iterations_used': max_iterations,
            'status': 'success'
        }
        
    except Exception as e:
        return {
            'pandas_code': None,
            'explanation': None,
            'judge_feedback': None,
            'validation_passed': False,
            'iterations_used': 0,
            'status': 'ERROR',
            'error_message': str(e)
        }

def _calculate_mathematical_breakdown(delivery_data: pd.Series, payout_rules: List[Dict]) -> str:
    """
    Generate detailed mathematical breakdown of payout calculation including fixed salary
    
    Args:
        delivery_data (pd.Series): Individual delivery partner's data
        payout_rules (List[Dict]): Payout rules configuration
        
    Returns:
        str: Mathematical formula and step-by-step calculation
    """
    try:
        calculation_steps = []
        running_total = 0

        name = delivery_data.get('Name', 'Unknown')
        zone = delivery_data.get('Delivery Zone', '')
        deliveries = delivery_data.get('Number of Deliveries', 0)
        distance = delivery_data.get('Distance travelled (km)', 0)
        full_shift = delivery_data.get('Full Shift (hours)', 0)
        peak_deliveries = delivery_data.get('Peak Hours Deliveries', 0)
        damages = delivery_data.get('Damage/Loss Items', 0)
        late_mins = delivery_data.get('Late Reporting (mins)', 0)
        advance = delivery_data.get('Advance/Salary Reimbursement (INR)', 0)
        fixed_salary = delivery_data.get('Fixed Salary (INR)', 0)
        
        # 1. Add fixed salary component
        if fixed_salary > 0:
            calculation_steps.append(f"Fixed Salary: ₹{fixed_salary}")
            running_total += fixed_salary
        
        # 2. Calculate delivery zone payout
        zone_rate = 0
        for rule in payout_rules:
            if rule['Category'] == 'Delivery Payout' and zone in rule['Condition']:
                amount_str = str(rule['Amount (₹)'])
                rate_match = re.search(r'₹(\d+)', amount_str)
                if rate_match:
                    zone_rate = int(rate_match.group(1))
                    break
        
        if zone_rate > 0 and deliveries > 0:
            delivery_payout = deliveries * zone_rate
            calculation_steps.append(f"Delivery Payout: {deliveries} deliveries × ₹{zone_rate} = ₹{delivery_payout}")
            running_total += delivery_payout
        
        # 3. Calculate distance payout
        if distance > 0:
            km_rate = 3  # Standard rate per kilometer
            distance_payout = distance * km_rate
            calculation_steps.append(f"Distance Payout: {distance} km × ₹{km_rate}/km = ₹{distance_payout}")
            running_total += distance_payout
        
        # 4. Calculate full shift bonus
        if full_shift > 0:
            shift_bonus = 100
            calculation_steps.append(f"Full Shift Bonus: ₹{shift_bonus}")
            running_total += shift_bonus
        
        # 5. Calculate peak hour bonus
        if peak_deliveries > 0:
            peak_rate = 5
            peak_bonus = peak_deliveries * peak_rate
            calculation_steps.append(f"Peak Hour Bonus: {peak_deliveries} deliveries × ₹{peak_rate} = ₹{peak_bonus}")
            running_total += peak_bonus
        
        # 6. Calculate damage/loss deductions
        if damages > 0:
            damage_penalty = damages * 100
            calculation_steps.append(f"Damage/Loss Penalty: {damages} items × ₹100 = -₹{damage_penalty}")
            running_total -= damage_penalty
        
        # 7. Calculate late reporting penalty
        if late_mins > 0:
            late_penalty = 100
            calculation_steps.append(f"Late Reporting Penalty: -₹{late_penalty}")
            running_total -= late_penalty
        
        # 8. Calculate salary advance deduction
        if advance > 0:
            calculation_steps.append(f"Salary Advance Deduction: -₹{advance}")
            running_total -= advance
        
        # Format final calculation with direct mathematical formula
        if calculation_steps:
            # Create direct calculation formula
            positive_components = []
            negative_components = []
            
            for step in calculation_steps:
                if " = -₹" in step or "Penalty:" in step or "Deduction:" in step:
                    # Extract negative amount
                    amount = step.split("₹")[1] if "₹" in step else "0"
                    negative_components.append(amount)
                else:
                    # Extract positive amount
                    if "₹" in step:
                        amount = step.split("₹")[1] if " = ₹" in step else step.split("₹")[1].split()[0]
                        positive_components.append(amount)
            
            # Build mathematical formula
            formula_parts = []
            if positive_components:
                formula_parts.append(" + ".join([f"₹{comp}" for comp in positive_components]))
            if negative_components:
                formula_parts.append(" - ".join([f"₹{comp}" for comp in negative_components]))
            
            mathematical_formula = " - ".join(formula_parts) if len(formula_parts) > 1 else formula_parts[0] if formula_parts else "₹0"
            
            breakdown = f"Mathematical Calculation for {name}:\n"
            breakdown += "\n".join([f"  {step}" for step in calculation_steps])
            breakdown += f"\n  Direct Calculation: {mathematical_formula} = ₹{running_total}"
            return breakdown
        
        return f"No applicable payout calculations found for {name}"
        
    except Exception as e:
        return f"Error in mathematical calculation: {str(e)}"


def execute_pandas_code(pandas_code: str) -> Dict[str, Any]:
    """
    Execute the generated pandas code safely using global dataframes and provide mathematical breakdown
    
    Example:
    Input: pandas_code="result = delivery_df.groupby('Delivery Zone')['Number of Deliveries'].sum()"
    Output: {
        'result': {'type': 'series', 'data': {'Zone A': 22, 'Zone B': 28}, 'summary': 'Returned series with 4 values'},
        'executed_code': 'result = delivery_df.groupby...',
        'mathematical_breakdown': ['Zone B: 28 deliveries × ₹20 = ₹560', ...],
        'calculation_explanation': 'Detailed step-by-step mathematical calculations',
        'status': 'success'
    }
    
    Args:
        pandas_code (str): Generated pandas code to execute
        
    Returns:
        Dict containing execution results, formatted data, mathematical breakdown, and execution status
    """
    global _delivery_df, _payout_df
    
    try:
        if _delivery_df is None or _payout_df is None:
            return {
                'result': None,
                'executed_code': pandas_code,
                'mathematical_breakdown': [],
                'calculation_explanation': '',
                'status': 'ERROR',
                'error_message': 'Dataframes not loaded. Please execute load_and_prepare_data first.'
            }

        safe_globals = {
            'pd': pd,
            'np': np,
            'delivery_df': _delivery_df,
            'payout_df': _payout_df,
            'result': None,
            're': re
        }

        exec(pandas_code, safe_globals)
        
        result = safe_globals.get('result')
        mathematical_breakdown = []
        calculation_explanation = ""

        if isinstance(result, pd.DataFrame):
            formatted_result = {
                'type': 'dataframe',
                'data': result.to_dict('records'),
                'columns': list(result.columns),
                'shape': result.shape,
                'summary': f"DataFrame with {result.shape[0]} rows and {result.shape[1]} columns"
            }

            if any(col.lower() in ['payout', 'total', 'earn', 'amount', 'salary'] for col in result.columns):
                payout_rules = _payout_df.to_dict('records')
                
                calculation_explanation = "Mathematical Calculation Breakdown:\n"
                
                for idx, row in result.iterrows():
                    if idx >= 3:
                        break
                    
                    name = row.get('Name', f'Record {idx+1}')

                    if 'Name' in row and name in _delivery_df['Name'].values:
                        delivery_data = _delivery_df[_delivery_df['Name'] == name].iloc[0]
                        math_breakdown = _calculate_mathematical_breakdown(delivery_data, payout_rules)
                        mathematical_breakdown.append(math_breakdown)

                    for col, val in row.items():
                        if isinstance(val, (int, float)) and col.lower() in ['payout', 'total', 'earn', 'amount']:
                            if val < 0:
                                mathematical_breakdown.append(f"{name}: Final Amount = ₹{val} (Negative value indicates net deduction)")
                            else:
                                mathematical_breakdown.append(f"{name}: Final Amount = ₹{val}")
                            break
                
                calculation_explanation = "Mathematical calculations performed based on payout rules and delivery data"
                            
        elif isinstance(result, pd.Series):
            formatted_result = {
                'type': 'series', 
                'data': result.to_dict(),
                'summary': f"Series with {len(result)} values"
            }

            total_value = result.sum() if result.dtype in ['int64', 'float64'] else 'N/A'
            calculation_explanation = f"Series calculation result. Total sum: {total_value}"
            
            for key, val in result.items():
                if isinstance(val, (int, float)):
                    if val < 0:
                        mathematical_breakdown.append(f"{key}: ₹{val} (Negative indicates net deduction)")
                    else:
                        mathematical_breakdown.append(f"{key}: ₹{val}")
        else:
            formatted_result = {
                'type': 'scalar',
                'data': str(result),
                'summary': f"Single value of type {type(result).__name__}"
            }
            
            if isinstance(result, (int, float)):
                calculation_explanation = f"Scalar calculation result: {result}"
                if result < 0:
                    mathematical_breakdown.append(f"Result: ₹{result} (Negative value indicates net deduction)")
                else:
                    mathematical_breakdown.append(f"Result: ₹{result}")
        
        return {
            'result': formatted_result,
            'executed_code': pandas_code,
            'mathematical_breakdown': mathematical_breakdown,
            'calculation_explanation': calculation_explanation,
            'status': 'success'
        }
        
    except Exception as e:
        return {
            'result': None,
            'executed_code': pandas_code,
            'mathematical_breakdown': [],
            'calculation_explanation': '',
            'status': 'ERROR', 
            'error_message': f"Code execution failed: {str(e)}"
        }

def format_response_with_calculations(query_result_json: str, user_query: str) -> Dict[str, Any]:
    """
    Format the analysis results with mathematical explanations and calculation methodology
    
    Example:
    Input: query_result_json='{"result": {"type": "dataframe", "data": [...]}}', user_query="Calculate total payouts"
    Output: {
        'formatted_response': {
            'query': 'Calculate total payouts',
            'results': {...},
            'mathematical_explanation': 'Calculations performed using delivery zone rates, distance...',
            'calculation_methodology': ['Zone-based delivery rates applied', 'Distance calculations...'],
            'data_summary': 'DataFrame with 4 rows and 3 columns'
        },
        'status': 'success'
    }
    
    Args:
        query_result_json (str): JSON string of results from pandas code execution
        user_query (str): Original user query for context
        
    Returns:
        Dict containing professionally formatted response with mathematical explanations
    """
    try:
        query_result = json.loads(query_result_json)
        
        if query_result['status'] != 'success':
            return {
                'formatted_response': None,
                'status': 'ERROR',
                'error_message': query_result.get('error_message', 'Query execution failed')
            }
        
        result_data = query_result['result']
        mathematical_breakdown = query_result.get('mathematical_breakdown', [])
        calculation_explanation = query_result.get('calculation_explanation', '')

        calculation_methodology = []
        
        if result_data['type'] == 'dataframe' and result_data['data']:
            df_data = result_data['data']
            columns = result_data.get('columns', [])

            if any('payout' in col.lower() for col in columns):
                calculation_methodology.append("Payout calculations based on zone-specific delivery rates")
                calculation_methodology.append("Distance-based compensation at ₹3 per kilometer")
                calculation_methodology.append("Performance bonuses for full shifts and peak hour deliveries")
                calculation_methodology.append("Penalty deductions for damages, losses, and late reporting")
                calculation_methodology.append("Salary advance adjustments applied to final amounts")
            
            if any('total' in col.lower() for col in columns):
                calculation_methodology.append("Aggregate totals computed across all applicable components")
            
            if any('zone' in col.lower() for col in columns):
                calculation_methodology.append("Zone-wise analysis with respective rate structures")

            if len(df_data) > 0:
                calculation_methodology.append(f"Analysis performed on {len(df_data)} records")
        
        elif result_data['type'] == 'series':
            calculation_methodology.append("Series-based calculation with aggregated values")
            data_values = list(result_data['data'].values())
            if data_values and all(isinstance(v, (int, float)) for v in data_values):
                total = sum(data_values)
                calculation_methodology.append(f"Total computed value: ₹{total:.2f}")

        mathematical_explanation = calculation_explanation
        if not mathematical_explanation and mathematical_breakdown:
            mathematical_explanation = "Mathematical calculations performed according to defined payout rules and delivery metrics"
        
        return {
            'formatted_response': {
                'query': user_query,
                'results': result_data,
                'mathematical_explanation': mathematical_explanation,
                'calculation_methodology': calculation_methodology,
                'mathematical_breakdown': mathematical_breakdown,
                'data_summary': result_data['summary']
            },
            'status': 'success'
        }
        
    except Exception as e:
        return {
            'formatted_response': None,
            'status': 'ERROR',
            'error_message': f"Response formatting failed: {str(e)}"
        }

# Tool 6: Complete analysis workflow with professional mathematical reporting
def analyze_delivery_data(csv_path: str, user_query: str) -> Dict[str, Any]:
    """
    Complete workflow: Load data, generate code, execute, and format response with mathematical breakdown
    
    Args:
        csv_path (str): Path to user's delivery CSV file
        user_query (str): Natural language query from user
        
    Returns:
        Dict containing complete analysis results with mathematical explanations and calculation methodology
    """
    try:

        data_context = load_and_prepare_data(csv_path)
        if data_context['status'] != 'success':
            return {
                'status': 'ERROR',
                'error_message': f"Data loading failed: {data_context.get('error_message', 'Unknown error')}"
            }

        code_result = generate_pandas_code(user_query, json.dumps(data_context))
        if code_result['status'] != 'success':
            return {
                'status': 'ERROR',
                'error_message': f"Code generation failed: {code_result.get('error_message', 'Unknown error')}"
            }
 
        execution_result = execute_pandas_code(code_result['pandas_code'])
        if execution_result['status'] != 'success':
            return {
                'status': 'ERROR',
                'error_message': f"Code execution failed: {execution_result.get('error_message', 'Unknown error')}"
            }
        
        final_result = format_response_with_calculations(
            json.dumps(execution_result), 
            user_query
        )
        
        if final_result['status'] != 'success':
            return {
                'status': 'ERROR',
                'error_message': f"Response formatting failed: {final_result.get('error_message', 'Unknown error')}"
            }

        final_result['validation_metadata'] = {
            'judge_feedback': code_result.get('judge_feedback', 'No feedback available'),
            'validation_passed': code_result.get('validation_passed', False),
            'iterations_used': code_result.get('iterations_used', 0),
            'generated_code': code_result.get('pandas_code', ''),
            'code_explanation': code_result.get('explanation', '')
        }
        
        return final_result
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'error_message': f"Complete analysis workflow failed: {str(e)}"
        }

analytics_processor = Agent(
    name="analytics_processor", 
    model="gemini-2.0-flash",
    description="Professional delivery analytics processor that interprets natural language queries and generates precise mathematical analysis using validated pandas code with comprehensive calculation breakdown",
    instruction="""
    You are a professional delivery analytics processor designed to provide precise, mathematically accurate analysis of delivery data. Your core competencies include:

    CORE CAPABILITIES:
    1. Automated data schema recognition and intelligent preprocessing
    2. Natural language query interpretation using advanced Gemini LLM processing
    3. Pandas code generation with iterative improvement through LLM judge validation (maximum 3 iterations)
    4. Secure code execution with comprehensive error handling
    5. Mathematical calculation breakdown with step-by-step explanations
    6. Professional response formatting with calculation methodology documentation
    7. Complete workflow automation from data ingestion to formatted results

    TECHNICAL SPECIFICATIONS:
    - Data Processing: Supports CSV format with intelligent column detection
    - Code Validation: Multi-iteration LLM judge system ensures code correctness
    - Mathematical Accuracy: Precise calculations based on defined payout rules
    - Security: Sandboxed code execution environment
    - Error Handling: Comprehensive exception management at all workflow stages

    PAYOUT CALCULATION FRAMEWORK:
    The system implements a comprehensive payout calculation framework based on the following mathematical models:

    1. FIXED SALARY COMPONENT:
       - Base fixed salary as specified in Fixed Salary (INR) column
       - Applied as the foundation amount for all calculations

    2. DELIVERY ZONE PAYOUTS:
       - Zone A: ₹25 per delivery
       - Zone B: ₹20 per delivery  
       - Zone C: ₹15 per delivery
       - Zone D: ₹15 per delivery

    3. DISTANCE-BASED COMPENSATION:
       - Rate: ₹3 per kilometer traveled
       - Applied to total distance covered

    4. PERFORMANCE BONUSES:
       - Full Shift Bonus: ₹100 per completed shift
       - Peak Hour Bonus: ₹5 per delivery during peak hours

    5. PENALTY DEDUCTIONS:
       - Damage/Loss Items: -₹100 per incident
       - Late Reporting: -₹100 per occurrence
       - Salary Advance: Deducted from total payout

    MATHEMATICAL METHODOLOGY:
    Total Payout = Fixed_Salary + (Deliveries × Zone_Rate) + (Distance × ₹3) + Shift_Bonus + Peak_Bonus - Penalties - Advances

    DIRECT CALCULATION FORMAT:
    For all mathematical calculations, provide direct formula showing:
    ₹[Fixed_Salary] + ₹[Delivery_Payout] + ₹[Distance_Payout] + ₹[Bonuses] - ₹[Penalties] - ₹[Deductions] = ₹[Total]

    RESPONSE FORMAT:
    All responses include:
    - Query execution results with proper data formatting
    - Step-by-step mathematical breakdown for calculations with direct formula
    - Calculation methodology explanation including fixed salary component
    - Validation metadata including judge feedback and iterations used
    - Professional formatting suitable for technical documentation

    QUALITY ASSURANCE:
    - Code validation through iterative LLM judge system
    - Mathematical accuracy verification including fixed salary component
    - Comprehensive error handling and status reporting
    - Professional documentation standards

    USAGE PROTOCOL:
    1. Load and validate data structure
    2. Parse natural language query intent
    3. Generate and validate pandas code
    4. Execute code with mathematical breakdown including fixed salary
    5. Format results with professional explanations and direct calculations
    6. Return comprehensive analysis report

    CALCULATION PRESENTATION:
    - Always show the direct mathematical formula
    - Include fixed salary as the base component
    - Present step-by-step breakdown with clear mathematical operations
    - Provide final result with complete calculation transparency

    Remember: You leverage Gemini's intelligence for code generation and LLM judge for validation, providing intelligent business analytics that helps users understand their delivery operations with guaranteed code quality and complete mathematical transparency including fixed salary components.
    """,
    tools=[
        load_and_prepare_data,
        judge_pandas_code,
        generate_pandas_code,
        execute_pandas_code,
        format_response_with_calculations,
        analyze_delivery_data,
    ]
)