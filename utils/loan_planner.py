from groq import Groq
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()
def data_input(principal, rate, time, income):
  loan_calc=compound_interest_calculation(principal, rate, time)
  emi_calc=emi_calculation(principal, rate, time, income)
  emi=emi_calc.get("EMI",0)
  check=financial_check(emi, income)
  metrics={"Loan_Amount":loan_calc.get("Amount",0),
          "Loan_Interest":loan_calc.get("Interest",0),
          "EMI":emi,
          "Net_take_income":emi_calc.get("Net_take_home",0),
          "Ratio":check.get("Ratio",0),
          "Zone":check.get("Zone",0)
          }
  advice=financial_advkce(metrics)
  
  return {"Loan_Amount":loan_calc.get("Amount",0),
          "Loan_Interest":loan_calc.get("Interest",0),
          "EMI":emi,
          "Net_take_income":emi_calc.get("Net_take_home",0),
          "Ratio":check.get("Ratio",0),
          "Zone":check.get("Zone",0),
          "Advice": advice
  }

def compound_interest_calculation(principal, rate, time):#rate per annum, time in years, amount in rupees
  amt=principal*((1+(rate/100))**time)
  interest=amt-principal
  return {"Amount":amt,"Interest":interest}

def emi_calculation(principal, rate, time, income):
  m_rate=rate/12
  m_time=time*12
  try:
    emi= (principal* m_rate/100 *(1+m_rate/100)**m_time)/(((1+m_rate/100)**m_time)-1)
  except Exception as e:
    emi=0
  net=income-emi
  return {"EMI":emi,"Net_take_home":net}
  
def financial_check(emi, income):
  ratio=(emi/income)*100
  if ratio<30:
    zone=1
  elif ratio <45:
    zone=0
  else:
    zone=-1
  return {"Ratio":ratio,"Zone":zone}

client = Groq(api_key=os.getenv("GROQ_API_KEY", "YOUR_API_KEY"))
message=data_input(principal, rate, time, income)

def financial_advice(message):
    try:
        user_prompt_string = json.dumps(message, indent=2)
        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a loan advisor, based on the info provided, give the insights whether the user should take a loan or not. What should they improve to make the loan suitable for them? Be precise and accurate."},
                {"role": "user", "content": f"Here are the financial metrics:{user_prompt_string}"}
            ]
        )

        return res.choices[0].message.content

    except Exception as e:
        print("🔥 GROQ ERROR:", e)  
        return "AI service is currently unavailable."

  
  
  
