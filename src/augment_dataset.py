"""
Dataset Augmentation — 100 India-Specific Phishing Messages
=============================================================
Adds 100 curated Indian phishing/scam messages to the dataset.
Covers all major India-specific lure categories:
  - Aadhaar / UIDAI scams        (15 messages)
  - KYC update scams             (12 messages)
  - UPI / payment scams          (15 messages)
  - Bank account freeze threats  (12 messages)
  - Government / IT dept scams   (12 messages)
  - Electricity / utility scams  (8 messages)
  - Loan / insurance scams       (10 messages)
  - Job / lottery scams          (8 messages)
  - Telecom / SIM scams          (8 messages)

All labelled as intent_label = 3 (Financial Phishing)

Usage:
    py -m src.augment_dataset
"""

import pandas as pd
from pathlib import Path

DATA_PATH   = Path("data/Bharat_Phish_Net_v1.csv")
BACKUP_PATH = Path("data/Bharat_Phish_Net_v1_backup.csv")


INDIA_PHISHING_100 = [

    # ── Aadhaar / UIDAI Scams (15) ────────────────────────────────────────
    "UIDAI Alert: Your Aadhaar card is linked to suspicious activity. Call 1800-XXX-XXXX immediately or your account will be permanently blocked.",
    "Your Aadhaar number requires urgent re-verification. Share the OTP received on your registered mobile to avoid deactivation of services.",
    "Dear citizen, your Aadhaar update is pending. Visit the link to submit biometric details within 24 hours or face service suspension.",
    "Aadhaar-linked bank account shows unauthorized login. Verify your PAN and Aadhaar number immediately at this link to secure your account.",
    "Your Aadhaar eKYC has expired. Update now to continue using government services. Click here and enter your Aadhaar OTP to proceed.",
    "UIDAI: Aadhaar card blocked due to multiple failed verification attempts. Call our helpline and provide your 12-digit Aadhaar to unblock.",
    "Alert: Your Aadhaar is being misused for a loan application. Verify your identity immediately by sharing OTP to cancel the application.",
    "Government notice: Aadhaar-PAN linking deadline has passed. Pay fine of Rs.1000 or link now using this link and your Aadhaar OTP.",
    "Your Aadhaar biometric is locked. Unlock it immediately by verifying OTP on this portal or your pension and subsidy will be stopped.",
    "UIDAI detected multiple login attempts on your Aadhaar profile. Secure your account by verifying your mobile OTP immediately.",
    "Aadhaar address update required. Submit your updated address proof and OTP within 48 hours to avoid Aadhaar deactivation.",
    "Your Aadhaar is linked to a suspicious bank transaction of Rs.49999. Call immediately and share OTP to block the transaction.",
    "mAadhaar App: Your virtual ID has expired. Generate new virtual ID by entering Aadhaar number and OTP on this link.",
    "UIDAI: Aadhaar seeding with your ration card is incomplete. Complete now to continue receiving food subsidy benefits.",
    "Final notice: Your Aadhaar has been flagged for duplicate registration. Verify by sharing your Aadhaar number and OTP within 24 hours.",

    # ── KYC Update Scams (12) ─────────────────────────────────────────────
    "Your KYC is incomplete. Bank account will be suspended within 24 hours. Update KYC immediately by clicking this link and entering details.",
    "Dear customer, KYC verification is mandatory as per RBI guidelines. Failure to update KYC will result in account freeze. Click here now.",
    "Your mobile KYC has expired. To continue using UPI services, update KYC immediately. Share OTP to verify your identity.",
    "RBI mandates updated KYC for all accounts. Your account shows pending KYC. Submit Aadhaar and PAN at this link to avoid blocking.",
    "Video KYC appointment scheduled. Join the link and keep your Aadhaar, PAN, and bank details ready for verification process.",
    "Alert: Your KYC documents are rejected. Resubmit your PAN card, Aadhaar number, and bank statement within 48 hours to avoid suspension.",
    "Paytm KYC: Your wallet KYC has expired. Your wallet will be blocked. Complete KYC by visiting store or clicking this link now.",
    "PhonePe KYC update required. Your UPI will stop working from midnight. Update KYC now to continue transactions without interruption.",
    "Your demat account KYC is incomplete. SEBI regulations require updated KYC. Submit documents at this link within 48 hours.",
    "CKYC update: Your central KYC registry needs updating. Update now to avoid suspension of banking and investment services.",
    "SBI KYC alert: Your savings account shows incomplete KYC. Visit branch or update online using Aadhaar OTP within 7 days.",
    "Last reminder: Complete your KYC for mutual fund account. SEBI will freeze account if KYC not done. Update at this link immediately.",

    # ── UPI / Payment Scams (15) ──────────────────────────────────────────
    "You have received a UPI payment of Rs.5000. Enter your UPI PIN to accept the payment into your account immediately.",
    "UPI cashback of Rs.1500 is pending in your account. Verify your UPI ID and enter PIN to receive the cashback right now.",
    "Your UPI ID has been temporarily blocked due to suspicious transaction. Call UPI helpline and share registered mobile OTP to unblock.",
    "Congratulations! You won Rs.10000 in UPI lucky draw. Enter your UPI PIN to claim the reward. Offer expires in 10 minutes.",
    "BHIM UPI Alert: Unauthorized transaction of Rs.9999 detected on your account. Call immediately and share OTP to reverse it.",
    "Your PhonePe account will be deactivated. Update UPI PIN and bank account details at this link to continue transactions.",
    "Google Pay: Transaction failed but Rs.2499 debited from your account. Share UPI PIN with our agent to process the refund.",
    "UPI payment of Rs.499 pending from government subsidy. Enter your UPI PIN on this link to receive the amount now.",
    "Your UPI daily transaction limit has been upgraded to Rs.2 lakh. Verify your PIN once to activate the new limit.",
    "Paytm: Rs.3000 cashback awarded on your recent transaction. Collect cashback by entering your UPI PIN on this secure link.",
    "NPCI Alert: Your UPI registration will expire tonight. Re-register by entering Aadhaar number and OTP to continue UPI services.",
    "UPI fraud detected on your account. Rs.15000 is being transferred. Enter PIN immediately on this link to block the transaction.",
    "Your UPI linked mobile number is being changed by someone. Cancel this request by verifying OTP on this link right now.",
    "Jio Money: Rs.750 reward points are expiring today. Redeem by linking your UPI account and entering PIN on this portal.",
    "IMPS transfer of Rs.24000 initiated from your account. If not done by you, call immediately and share OTP to reverse.",

    # ── Bank Account Freeze Threats (12) ──────────────────────────────────
    "SBI Alert: Your account has been frozen due to suspicious activity. Call 1800-XXX-XXXX and share account number and OTP to unfreeze.",
    "Dear HDFC customer, your debit card has been blocked. To unblock, verify card details and CVV at this secure link immediately.",
    "Your bank account shows unauthorized login from unknown device. Share OTP immediately to reverse the suspicious transaction.",
    "ICICI Bank: Rs.24999 deducted from your account. If not done by you, call immediately and share card details to raise dispute.",
    "Your account will be closed within 12 hours due to suspicious transactions. Login and verify identity at link to prevent closure.",
    "Axis Bank: Your net banking has been disabled. Click the link and enter your login ID, password, and OTP to re-enable access.",
    "Kotak Bank: Your account shows 3 failed login attempts. Account will be locked. Verify by entering OTP on this link now.",
    "Your fixed deposit of Rs.1.5 lakh will be broken automatically to recover dues. Call immediately to dispute and share account OTP.",
    "Bank of Baroda: Your account linked Aadhaar has a mismatch. Update Aadhaar details within 24 hours to avoid account suspension.",
    "Yes Bank: Suspicious outward NEFT of Rs.49000 pending. If not authorized, click this link and enter OTP to cancel immediately.",
    "PNB Alert: Your account has been flagged under PMLA. Verify KYC and Aadhaar details within 48 hours to restore account access.",
    "Canara Bank: Your account will be deactivated due to inactivity. Reactivate by verifying Aadhaar OTP at this link within today.",

    # ── Government / Income Tax / EPF Scams (12) ─────────────────────────
    "Income Tax Dept: You have outstanding tax refund of Rs.8500. Submit PAN and bank account number at this link to receive it now.",
    "GST department notice: Your business has pending GST dues. Pay immediately using this link or face legal action and account seizure.",
    "PM relief fund is providing Rs.3000 to eligible citizens. Register with Aadhaar and bank details at this portal to receive funds.",
    "IT department notice issued against your PAN. Respond within 24 hours or account will be attached. Call helpline with PAN details.",
    "EPF withdrawal of Rs.15000 pending in your account. Submit bank account details and Aadhaar OTP to release the amount now.",
    "EPFO: Your PF account shows unclaimed balance of Rs.32000. Claim now by submitting Aadhaar and bank details on the portal.",
    "Court notice: A case has been filed against your Aadhaar number for tax evasion. Call this number immediately to settle.",
    "Cyber Crime Cell: Your mobile number is linked to fraudulent activity. Call immediately or face arrest within 24 hours.",
    "Ministry of Finance: COVID relief payment of Rs.5000 is pending for your Aadhaar. Submit bank details to receive it today.",
    "PM Kisan: Your installment of Rs.2000 is on hold due to bank account mismatch. Update account details linked to Aadhaar now.",
    "ESIC benefit of Rs.8500 unclaimed in your account. Claim within 7 days by submitting Aadhaar, PAN, and bank details at this link.",
    "Government scholarship of Rs.20000 approved for your ward. Submit Aadhaar and bank details within 48 hours to disburse amount.",

    # ── Electricity / Utility Scams (8) ──────────────────────────────────
    "BESCOM Alert: Your electricity connection will be disconnected today at 9 PM due to unpaid dues. Pay immediately to avoid disconnection.",
    "Your electricity bill of Rs.1842 is overdue. Pay now using this link or connection will be cut within 2 hours permanently.",
    "MSEDCL notice: Last warning for unpaid electricity dues. Pay Rs.2150 within 1 hour to avoid disconnection and legal action.",
    "Your gas connection will be suspended due to pending KYC. Complete gas KYC by entering Aadhaar details on this link today.",
    "TATA Power: Due to technical issue your electricity bill of Rs.1500 not received. Pay now on this link to avoid disconnection.",
    "Electricity department: Smart meter installation requires your Aadhaar verification. Share OTP to schedule the installation now.",
    "Your water connection will be disconnected due to non-payment of Rs.890. Pay using UPI on this link within 3 hours today.",
    "IGL gas bill of Rs.2340 unpaid for 2 months. Pay immediately on this link or gas supply will be disconnected tonight.",

    # ── Loan / Insurance Scams (10) ───────────────────────────────────────
    "Pre-approved personal loan of Rs.2 lakh in your name. Pay Rs.999 processing fee now to receive the amount within 2 hours.",
    "IRDAI alert: Your insurance claim of Rs.50000 is approved. Pay stamp duty of Rs.500 to release your claim amount immediately.",
    "Your home loan application is approved. Pay processing fee of Rs.1500 to receive disbursement within 24 hours to your account.",
    "Easy loan of Rs.50000 approved instantly. No CIBIL check required. Pay Rs.199 registration and provide Aadhaar, PAN to disburse.",
    "LIC policy bonus of Rs.35000 due for maturity. Submit policy number, Aadhaar, and bank details at this link to receive amount.",
    "Your car insurance has expired. Renew now for Rs.1999 only. Click link and share card details for instant policy renewal.",
    "PMJJBY insurance claim of Rs.2 lakh approved for your nominee. Submit nominee Aadhaar and bank details to process payment.",
    "RBI approved lender: Rs.3 lakh instant loan approved for your profile. Pay Rs.500 token amount to lock the offer today.",
    "Your mutual fund SIP payment failed. Update your bank mandate by sharing account details and OTP to avoid SIP cancellation.",
    "Term insurance payout of Rs.10 lakh pending for your claim. Pay agent commission of Rs.2000 to release the claim amount.",

    # ── Job / Lottery Scams (8) ───────────────────────────────────────────
    "TCS/Infosys HR: Your application is selected for joining. Pay Rs.1500 security deposit to confirm joining and receive offer letter.",
    "KBC Lottery: Your mobile number won Rs.25 lakh in KBC lucky draw. Contact Mr. Amitabh on WhatsApp to claim your prize now.",
    "Jio lottery: Your mobile number won Rs.2 lakh prize. Share your bank account number and OTP to receive the prize money.",
    "Work from home job offer: Earn Rs.15000 per month by spending 2 hours daily. Pay Rs.500 registration to start working today.",
    "Amazon work from home: Part time job approved for you. Pay Rs.299 to receive the starter kit and begin earning immediately.",
    "Railway recruitment board: Your application is shortlisted for Group D. Pay Rs.500 examination fee at this link to confirm.",
    "Your resume has been shortlisted by MNC company. Pay Rs.999 background verification fee to proceed with the interview process.",
    "International lottery: Your email ID won USD 500000. Pay processing fee of Rs.5000 to receive the converted amount today.",

    # ── Telecom / SIM Scams (8) ───────────────────────────────────────────
    "Your Airtel SIM will be blocked within 24 hours due to incomplete KYC. Complete SIM KYC by sharing Aadhaar OTP immediately.",
    "Jio Alert: Your number is scheduled for deactivation. Recharge with Rs.199 plan using this link and share OTP to continue service.",
    "BSNL: Your SIM card requires re-verification due to new TRAI guidelines. Share Aadhaar details and OTP to avoid disconnection.",
    "Vi (Vodafone-Idea): Your postpaid bill is overdue. Pay Rs.1240 now using this link or your outgoing calls will be barred tonight.",
    "TRAI notice: Your mobile number is linked to cybercrime activity. Call this number in 2 hours or SIM will be blocked by DOT.",
    "Your phone number is being ported without your consent. Cancel porting request immediately by sharing OTP received on your mobile.",
    "Airtel broadband: Your connection will be disconnected due to Fair Usage Policy violation. Pay upgrade fee at this link now.",
    "DOT order: Your mobile number will be disconnected in 2 hours for misuse. Call the given number immediately with Aadhaar details.",
]


def augment_dataset():
    print(f"Loading existing dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"Original dataset size: {len(df)} messages")

    # Backup
    df.to_csv(BACKUP_PATH, index=False)
    print(f"Backup saved → {BACKUP_PATH}")

    print(f"\nLabel distribution BEFORE:")
    label_names = {0: "Legitimate", 1: "Promo Spam", 2: "Generic Scam", 3: "Financial Phishing"}
    for label, count in df["intent_label"].value_counts().sort_index().items():
        print(f"  Label {label} ({label_names.get(label, '?')}): {count}")

    # Build new rows
    new_rows = []
    for msg in INDIA_PHISHING_100:
        row = {col: None for col in df.columns}
        row["message"]      = msg
        row["clean_text"]   = msg.lower()
        row["intent_label"] = 3
        new_rows.append(row)

    new_df       = pd.DataFrame(new_rows)
    augmented_df = pd.concat([df, new_df], ignore_index=True)
    augmented_df.to_csv(DATA_PATH, index=False)

    print(f"\nLabel distribution AFTER:")
    for label, count in augmented_df["intent_label"].value_counts().sort_index().items():
        print(f"  Label {label} ({label_names.get(label, '?')}): {count}")

    print(f"\nAdded: {len(INDIA_PHISHING_100)} India-specific phishing messages")
    print(f"Label 3 grew: 13 → {13 + len(INDIA_PHISHING_100)}")
    print(f"New total: {len(augmented_df)} messages")
    print(f"Saved → {DATA_PATH}")

    print("\n--- Next Steps ---")
    print("  py -m src.layer2_model      (retrain)")
    print("  py -m src.extract_concepts  (re-extract SHAP vocabulary)")
    print("  py -m src.cluster_concepts  (re-cluster)")
    print("  py -m src.explain_shap      (test Aadhaar/KYC/UPI messages)")


if __name__ == "__main__":
    assert len(INDIA_PHISHING_100) == 100, f"Expected 100 messages, got {len(INDIA_PHISHING_100)}"
    augment_dataset()
