# Delivery Analytics Agent 

An AI agent built using **Google ADK** that enables seamless analysis of delivery personnel data. Users can interact using natural language to retrieve insights, validate conditions, and calculate payouts, bonuses, and more — all with transparency and LLM-based reasoning.

#  Overview

This agent processes two CSV files:
1. **Delivery Data** – Contains delivery person details such as:
   - Name
   - Number of deliveries
   - Number of returns
   - Zone(s) of operation

2. **Payout Rules** – Specifies:
   - Payouts by zone
   - Full shift bonuses
   - Delivery caps, penalties, and more

With these, users can ask questions like:
- _"What is the total salary for John?"_
- _"How much bonus did agents in Zone A receive?"_
- _"Why is Mary’s salary lower than others?"_

## Features

- ⚡Built using **Google ADK** for LLM-powered agent creation
-  **Prompt caching** to avoid redundant LLM calls
-  **Supports complex **payout and bonus logic**
-  **Step-by-step calculation breakdowns** for transparency
-  **LLM-based validation judge** that refines and approves user queries before execution

##  Input CSV Formats

### 1. `deliveries.csv`
| Name | Zone | Deliveries | Returns |
|------|------|------------|---------|
| John | A    | 45         | 2       |

### 2. `payout_rules.csv`
| Zone | Payout per Delivery | Full Shift Bonus |
|------|----------------------|------------------|
| A    | 20                   | 200              |

##  Example Queries

- "Show me how much salary Rahul should get."
- "Break down the payout for Zone B deliveries."
- "Why did Akash not qualify for full shift bonus?"

##  Running the Agent

```bash
adk web
