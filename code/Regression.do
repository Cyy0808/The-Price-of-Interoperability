/* ====================================================================
   Replication Code
   Paper: The Price of Interoperability: Exploring Cross-Chain Bridges
          and Their Economic Consequences
   
   Structure:
     Part 0  -- Setup and prerequisites
     Part 1  -- Table 1:  Drivers of net capital inflow
     Part 2  -- Table 2:  Impact of ASI and AAI on economic growth
     Part 3  -- Table 3:  Term structure of impact on native token returns
	 Part 4  -- Table 4:  Impact of interoperability on congestion and costs
     Part 5  -- Table 5:  Impact of Multichain collapse (DiD)
     Part 6  -- Table 6:  Impact of interoperability on TVL correlation
     Part 7  -- Table 7:  Heterogeneity by chain type (EVM/L1)
     Part 8  -- Table 8:  Heterogeneity by bridge type (Official/Third-party)
     Part 9  -- Table 9:  Heterogeneity by cross-chain mechanism (LNM/BNM/LP)
     Appendix -- Correlation matrix and auxiliary regressions
   ==================================================================== */


* =====================================================================
* Part 0: Setup
* =====================================================================

clear all
set more off
macro drop _all
estimates clear


* =====================================================================
* Part 1: Table 1 -- Drivers of Net Capital Inflow
* =====================================================================

use "Sample_Cleaned.dta", clear
xtset chain_id date_common

// Construct dependent variable: future 7-day average net inflow (signed log)
tssmooth ma temp_F7_flow = net_flow_usd, window(0 1 6)
gen ln_F7_NetInflow = sign(temp_F7_flow) * log(abs(temp_F7_flow) + 1)
drop temp_F7_flow

// Construct current-day token return
gen Current_Ret = D.ln_price

estimates clear

// Model (1): Return + economic controls + ASI
quietly xtreg D.ln_F7_NetInflow Current_Ret ln_tvl ln_dau ln_contracts ///
    ASI AAI i.date_common, fe vce(cluster chain_id)
estimates store tab1_m1

// Model (2): Full specification with gas cost controls
quietly xtreg D.ln_F7_NetInflow Current_Ret ln_tvl ln_dau ln_contracts ///
    ASI AAI ln_gas_fee_per_tx ln_total_gas_usd ln_gas_used ///
    i.date_common, fe vce(cluster chain_id)
estimates store tab1_m2

// Export
esttab tab1_m1 tab1_m2, ///
    b(3) t(2) star(* 0.1 ** 0.05 *** 0.01) ///
    keep(Current_Ret ln_tvl ln_dau ln_contracts ASI AAI ///
         ln_gas_fee_per_tx ln_total_gas_usd ln_gas_used) ///
    mtitles("(1)" "(2)") ///
    stats(N r2_w, labels("Observations" "R-squared")) ///
    title("Table 1: Drivers of Net Capital Inflow")


* =====================================================================
* Part 2: Table 2 -- Impact of ASI and AAI on Economic Growth
* =====================================================================

use "Sample_Cleaned.dta", clear
xtset chain_id date_common
estimates clear

// --- Panel A: ASI ---
quietly xtreg F7_ln_tvl ASI ///
    ln_dau ln_contracts F1_Ret ln_total_gas_usd i.date_common, fe
estimates store tab2a_tvl

quietly xtreg F7_ln_dau ASI ///
    ln_tvl ln_contracts F1_Ret ln_total_gas_usd i.date_common, fe
estimates store tab2a_dau

quietly xtreg F7_ln_contracts ASI ///
    ln_tvl ln_dau F1_Ret ln_total_gas_usd i.date_common, fe
estimates store tab2a_cont

// --- Panel B: AAI ---
quietly xtreg F7_ln_tvl AAI ///
    ln_dau ln_contracts F1_Ret ln_total_gas_usd ln_tvl i.date_common, fe
estimates store tab2b_tvl

quietly xtreg F7_ln_dau AAI ///
    ln_tvl ln_contracts F1_Ret ln_total_gas_usd i.date_common, fe
estimates store tab2b_dau

quietly xtreg F7_ln_contracts AAI ///
    ln_total_gas_usd F1_Ret i.date_common, fe
estimates store tab2b_cont

// Export Panel A
esttab tab2a_tvl tab2a_dau tab2a_cont, ///
    b(3) t(2) star(* 0.1 ** 0.05 *** 0.01) ///
    keep(ASI) ///
    mtitles("TVL (MA7)" "DAU (MA7)" "New Contracts (MA7)") ///
    stats(N r2_w, labels("Observations" "R-squared")) ///
    title("Table 2 Panel A: Impact of ASI on Economic Growth")

// Export Panel B
esttab tab2b_tvl tab2b_dau tab2b_cont, ///
    b(3) t(2) star(* 0.1 ** 0.05 *** 0.01) ///
    keep(AAI) ///
    mtitles("TVL (MA7)" "DAU (MA7)" "New Contracts (MA7)") ///
    stats(N r2_w, labels("Observations" "R-squared")) ///
    title("Table 2 Panel B: Impact of AAI on Economic Growth")


* =====================================================================
* Part 3: Table 3 -- Term Structure of Impact on Native Token Returns
* =====================================================================

use "Sample_Cleaned.dta", clear
xtset chain_id date_common
estimates clear

foreach k in 1 3 7 15 30 100 {
    quietly xtreg F`k'_Ret ASI AAI ///
        tvl ln_dau ln_contracts ln_total_gas_usd L2.F1_Ret ///
        i.date_common, fe
    estimates store tab3_`k'd
}

esttab tab3_1d tab3_3d tab3_7d tab3_15d tab3_30d tab3_100d, ///
    b(3) t(2) star(* 0.1 ** 0.05 *** 0.01) ///
    keep(ASI AAI) ///
    mtitles("1-Day" "3-Day" "7-Day" "15-Day" "30-Day" "100-Day") ///
    stats(N r2_w, labels("Observations" "R-squared")) ///
    title("Table 3: Term Structure of Impact on Native Token Returns")


* =====================================================================
* Part 4: Table 4 -- Impact of Interoperability on Congestion and Costs
* =====================================================================

use "Sample_Cleaned.dta", clear
xtset chain_id date_common
estimates clear

quietly xtreg F7_ln_total_gas_usd ASI AAI ///
    ln_tvl ln_dau ln_contracts F1_Ret i.date_common, fe
estimates store tab6_fee

quietly xtreg F7_ln_gas_used ASI AAI ///
    ln_dau ln_contracts F1_Ret i.date_common, fe
estimates store tab6_used

quietly xtreg F7_ln_gas_fee_per_tx ASI AAI ///
    ln_tvl ln_dau ln_contracts F1_Ret i.date_common, fe
estimates store tab6_avg

esttab tab6_fee tab6_used tab6_avg, ///
    b(3) t(2) star(* 0.1 ** 0.05 *** 0.01) ///
    keep(ASI AAI) ///
    mtitles("Gas Fee" "Gas Used" "Avg Gas/Tx") ///
    stats(N r2_w, labels("Observations" "R-squared")) ///
    title("Table 6: Impact of Interoperability on Congestion and Costs")

	
* =====================================================================
* Part 5: Table 5 -- Impact of Multichain Collapse on Chain Economy (DiD)
* =====================================================================

use "Sample_Cleaned.dta", clear
duplicates drop chain_id date_common, force
xtset chain_id date_common
estimates clear

// Define treatment: chains supported by Multichain
gen treat = 0
replace treat = 1 if inlist(chain, "avalanche", "cronos", "ethereum", ///
    "optimism", "arbitrum", "polygon", "rootstock", "sonic")

// Define post period
gen post = (date_common >= date("2023-07-06", "YMD"))

// DiD interaction
gen did = treat * post

// Regression with controls
quietly xtreg ln_tvl did ///
    ln_dau F1_Ret ln_total_gas_usd i.date_common, fe
estimates store tab4_did

esttab tab4_did, ///
    b(3) t(2) star(* 0.1 ** 0.05 *** 0.01) r2 ///
    keep(did) ///
    mtitles("Log TVL") ///
    stats(N r2_w, labels("Observations" "R-squared")) ///
    title("Table 4: Impact of Multichain Collapse (DiD)")


* =====================================================================
* Part 6: Table 6 -- Impact of Interoperability on TVL Correlation
*
* Note: This section constructs a dyadic (chain-pair × day) dataset
*       from scratch using pairwise_distances.csv and bridge flows.
* =====================================================================

// --- Step 6.1: Construct PSI (Pairwise Structural Interoperability) ---
import delimited "pairwise_distances.csv", clear

gen date_common = date(date, "YMD")
format date_common %td
drop date

foreach v in source_chain target_chain {
    replace `v' = trim(lower(`v'))
    replace `v' = "rootstock" if `v' == "rootstock_rsk"
}

// Undirected pair ordering
gen chain_1 = cond(source_chain < target_chain, source_chain, target_chain)
gen chain_2 = cond(source_chain < target_chain, target_chain, source_chain)

gen inv_dist = cond(distance >= ., 0, 1/distance)
collapse (sum) sum_inv_dist = inv_dist, by(chain_1 chain_2 date_common)
gen PSI = sum_inv_dist / 2
label var PSI "Pairwise Structural Interoperability"

// Filter to sample chains
foreach c in chain_1 chain_2 {
    keep if inlist(`c', "aptos", "arbitrum", "avalanche", "base", "bnb") | ///
            inlist(`c', "cronos", "ethereum", "hyperliquid", "linea", "opbnb") | ///
            inlist(`c', "optimism", "polygon", "rootstock", "sei", "solana") | ///
            inlist(`c', "sonic", "sui", "tron", "unichain")
}

sort chain_1 chain_2 date_common
save "temp_pair_PSI.dta", replace

// --- Step 6.2: Construct bilateral flow (X2) ---
import delimited "merged_cleaned_result.csv", clear

gen date_common = date(date, "YMD")
format date_common %td

replace source_chain      = "rootstock" if source_chain == "rootstock_rsk"
replace destination_chain = "rootstock" if destination_chain == "rootstock_rsk"
replace total_amount_usd  = avg_transfer_usd_value * transfer_count ///
    if total_amount_usd == .

// Undirected pair
gen chain_1 = cond(source_chain < destination_chain, ///
    source_chain, destination_chain)
gen chain_2 = cond(source_chain < destination_chain, ///
    destination_chain, source_chain)

collapse (sum) pair_total_flow = total_amount_usd, ///
    by(chain_1 chain_2 date_common)

// Filter chains
foreach c in chain_1 chain_2 {
    keep if inlist(`c', "aptos", "arbitrum", "avalanche", "base", "bnb") | ///
            inlist(`c', "cronos", "ethereum", "hyperliquid", "linea", "opbnb") | ///
            inlist(`c', "optimism", "polygon", "rootstock", "sei", "solana") | ///
            inlist(`c', "sonic", "sui", "tron", "unichain")
}

sort chain_1 chain_2 date_common
save "temp_pair_flow.dta", replace

// --- Step 6.3: Construct rolling TVL correlations (Y) ---
use "Sample_Cleaned.dta", clear
drop if missing(chain_id) | missing(date_common)
xtset chain_id date_common
gen tvl_growth = D.ln_tvl
keep chain chain_id date_common tvl_growth ln_tvl ln_total_gas_usd ASI
save "temp_node_data.dta", replace

// Build pair skeleton
use "temp_node_data.dta", clear
keep chain
duplicates drop
rename chain chain_1
save "temp_list_c1.dta", replace
rename chain_1 chain_2
save "temp_list_c2.dta", replace

use "temp_list_c1.dta", clear
cross using "temp_list_c2.dta"
keep if chain_1 < chain_2
save "temp_pair_list.dta", replace

use "Sample_Cleaned.dta", clear
keep date_common
duplicates drop
cross using "temp_pair_list.dta"
sort chain_1 chain_2 date_common
save "temp_pair_skeleton.dta", replace

// Merge node-level data
use "temp_pair_skeleton.dta", clear

rename chain_1 chain
merge m:1 chain date_common using "temp_node_data.dta", ///
    keep(match master) nogenerate
rename (tvl_growth ln_tvl ln_total_gas_usd ASI) ///
    (tvl_growth_1 ln_tvl_1 ln_gas_1 ASI_1)
rename chain chain_1

rename chain_2 chain
merge m:1 chain date_common using "temp_node_data.dta", ///
    keep(match master) nogenerate
rename (tvl_growth ln_tvl ln_total_gas_usd ASI) ///
    (tvl_growth_2 ln_tvl_2 ln_gas_2 ASI_2)
rename chain chain_2

egen pair_id = group(chain_1 chain_2)
sort pair_id date_common

// Rolling correlations
foreach w in 30 60 90 {
    capture drop Y_Corr_TVL_`w' corr_* reg_*
    rangestat (corr) tvl_growth_1 tvl_growth_2, ///
        interval(date_common -`w' 0) by(pair_id)
    
    capture drop corr_nobs
    capture drop reg_n
    rename corr_* Y_Corr_TVL_`w'
}

winsor2 Y_Corr_TVL_30 Y_Corr_TVL_60 Y_Corr_TVL_90, cuts(1 99) replace

save "temp_pair_Y.dta", replace

// --- Step 6.4: Merge X and Y, then run regressions ---
use "temp_pair_Y.dta", clear

// Merge PSI
merge 1:1 chain_1 chain_2 date_common using "temp_pair_PSI.dta"
drop if _merge == 2
drop _merge

// Merge bilateral flow
merge 1:1 chain_1 chain_2 date_common using "temp_pair_flow.dta"
drop if _merge == 2
drop _merge
replace pair_total_flow = 0 if pair_total_flow == .

// Deduplicate and set panel before tssmooth
duplicates drop chain_1 chain_2 date_common, force
capture drop pair_id
egen pair_id = group(chain_1 chain_2)
xtset pair_id date_common

// Flow variable: log + 7-day moving average
gen ln_pair_flow = log(pair_total_flow + 1)
tssmooth ma X_Flow_MA7 = ln_pair_flow, window(6 1 0)

// Control variable
gen Control_Diff_TVL = abs(ln_tvl_1 - ln_tvl_2) / (ln_tvl_1 + ln_tvl_2)

save "Final_Dyadic_Dataset.dta", replace

estimates clear

foreach w in 30 60 90 {
    quietly xtreg Y_Corr_TVL_`w' PSI X_Flow_MA7 ///
        Control_Diff_TVL i.date_common, fe
    estimates store tab5_`w'd
}

esttab tab5_30d tab5_60d tab5_90d, ///
    b(3) t(2) star(* 0.1 ** 0.05 *** 0.01) ///
    keep(PSI X_Flow_MA7) ///
    mtitles("30-Day" "60-Day" "90-Day") ///
    stats(N r2_w, labels("Observations" "R-squared")) ///
    title("Table 5: Impact of Interoperability on TVL Correlation")

// Clean up temp files
foreach f in pair_PSI pair_flow node_data ///
    list_c1 list_c2 pair_list pair_skeleton pair_Y {
    capture erase "temp_`f'.dta"
}



* =====================================================================
* Part 7: Table 7 -- Heterogeneity by Chain Type (EVM × Layer)
*         (Appendix Table: full regression with interaction terms)
* =====================================================================

use "Sample_Cleaned.dta", clear
xtset chain_id date_common
estimates clear

// Generate interaction terms
gen ASI_isEVM = ASI * isEVM
gen ASI_isL1  = ASI * isL1
gen AAI_isEVM = AAI * isEVM
gen AAI_isL1  = AAI * isL1

// --- ASI interactions ---
quietly xtreg F7_ln_tvl ASI ASI_isL1 ASI_isEVM AAI AAI_isEVM AAI_isL1 ///
    ln_dau ln_contracts F1_Ret ln_total_gas_usd i.date_common, fe
estimates store tab7_asi_tvl

quietly xtreg F7_ln_gas_fee_per_tx ASI ASI_isL1 ASI_isEVM AAI AAI_isEVM AAI_isL1 ///
    ln_tvl ln_dau ln_contracts F1_Ret i.date_common, fe
estimates store tab7_asi_gas

// --- AAI interactions ---
quietly xtreg F7_ln_tvl AAI AAI_isEVM AAI_isL1 ///
    ln_dau ln_contracts F1_Ret ln_total_gas_usd i.date_common, fe
estimates store tab7_aai_tvl

quietly xtreg F7_ln_gas_fee_per_tx AAI AAI_isEVM AAI_isL1 ///
    ln_tvl ln_dau ln_contracts F1_Ret i.date_common, fe
estimates store tab7_aai_gas

esttab tab7_asi_tvl tab7_asi_gas, ///
    b(3) t(2) star(* 0.1 ** 0.05 *** 0.01) ///
    keep(ASI ASI_isL1 ASI_isEVM AAI AAI_isEVM AAI_isL1) ///
    mtitles("TVL (ASI)" "Gas (ASI)" "TVL (AAI)" "Gas (AAI)") ///
    stats(N r2_w, labels("Observations" "R-squared")) ///
    title("Table 7: Heterogeneity by Chain Type")


* =====================================================================
* Part 8: Table 8 -- Heterogeneity by Bridge Type (Official vs Third-Party)
* =====================================================================

use "Sample_Cleaned.dta", clear
xtset chain_id date_common
estimates clear

// --- Panel A: ASI by bridge type ---
quietly xtreg F7_ln_tvl ASI_Offi ASI_TP AAI ///
    ln_dau ln_contracts F1_Ret ln_total_gas_usd i.date_common, fe
estimates store tab8a_tvl

quietly xtreg F7_ln_gas_fee_per_tx ASI_Offi ASI_TP AAI ///
    ln_tvl ln_dau ln_contracts F1_Ret i.date_common, fe
estimates store tab8a_gas

// --- Panel B: AAI by bridge type ---
quietly xtreg F7_ln_tvl ASI AAI_offi AAI_tp ///
    ln_dau ln_contracts F1_Ret ln_total_gas_usd i.date_common, fe
estimates store tab8b_tvl

quietly xtreg F7_ln_gas_fee_per_tx ASI AAI_offi AAI_tp ///
    ln_tvl ln_dau ln_contracts F1_Ret i.date_common, fe
estimates store tab8b_gas

// Export Panel A
esttab tab8a_tvl tab8a_gas, ///
    b(3) t(2) star(* 0.1 ** 0.05 *** 0.01) ///
    keep(ASI_Offi ASI_TP) ///
    mtitles("TVL" "Avg Gas/Tx") ///
    stats(N r2_w, labels("Observations" "R-squared")) ///
    title("Table 8 Panel A: ASI by Bridge Type")

// Export Panel B
esttab tab8b_tvl tab8b_gas, ///
    b(3) t(2) star(* 0.1 ** 0.05 *** 0.01) ///
    keep(AAI_offi AAI_tp) ///
    mtitles("TVL" "Avg Gas/Tx") ///
    stats(N r2_w, labels("Observations" "R-squared")) ///
    title("Table 8 Panel B: AAI by Bridge Type")


* =====================================================================
* Part 9: Table 9 -- Heterogeneity by Cross-Chain Mechanism
* =====================================================================

use "Sample_Cleaned.dta", clear
xtset chain_id date_common
estimates clear

// --- Panel A: ASI by mechanism ---
quietly xtreg F7_ln_tvl ASI_LNM ASI_BNM ASI_LP AAI ///
    ln_dau ln_contracts F1_Ret ln_total_gas_usd i.date_common, fe
estimates store tab9a_tvl

quietly xtreg F7_ln_gas_fee_per_tx ASI_LNM ASI_BNM ASI_LP AAI ///
    ln_tvl ln_dau ln_contracts F1_Ret i.date_common, fe
estimates store tab9a_gas

// --- Panel B: AAI by mechanism ---
quietly xtreg F7_ln_tvl ASI AAI_lnm AAI_bnm AAI_lp ///
    ln_tvl ln_dau ln_contracts F1_Ret ln_total_gas_usd i.date_common, fe
estimates store tab9b_tvl

quietly xtreg F7_ln_gas_fee_per_tx ASI AAI_lnm AAI_bnm AAI_lp ///
    ln_tvl ln_dau ln_contracts F1_Ret i.date_common, fe
estimates store tab9b_gas

// Export Panel A
esttab tab9a_tvl tab9a_gas, ///
    b(3) t(2) star(* 0.1 ** 0.05 *** 0.01) ///
    keep(ASI_LNM ASI_BNM ASI_LP) ///
    mtitles("TVL" "Avg Gas/Tx") ///
    stats(N r2_w, labels("Observations" "R-squared")) ///
    title("Table 9 Panel A: ASI by Mechanism")

// Export Panel B
esttab tab9b_tvl tab9b_gas, ///
    b(3) t(2) star(* 0.1 ** 0.05 *** 0.01) ///
    keep(AAI_lnm AAI_bnm AAI_lp) ///
    mtitles("TVL" "Avg Gas/Tx") ///
    stats(N r2_w, labels("Observations" "R-squared")) ///
    title("Table 9 Panel B: AAI by Mechanism")


* =====================================================================
* Appendix: Correlation Matrix
* =====================================================================

use "Sample_Cleaned.dta", clear

pwcorr ASI AAI ln_tvl ln_dau ln_contracts, star(0.01) sig


