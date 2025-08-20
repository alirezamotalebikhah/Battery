

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pyomo.environ import *


#define model
model = ConcreteModel(doc="Home energy Management")
#define Set
model.T=RangeSet(1,96,doc="Time Period")
model.A = RangeSet(1, 9, doc="Appliance")
dt=0.25
#Define Parameters
#Appilince Data
appiliance = {
    1: {"name": "Dishwasher", "ratedpower": 2.8, "preferd": range(81, 87), "shift": range(61, 87), "must": 6},
    2: {"name": "Washing", "ratedpower": 3.5, "preferd": range(40, 43), "shift": range(37, 45), "must": 3},
    3: {"name": "Pump", "ratedpower": 1.5, "preferd": range(41, 53), "shift": range(25, 53), "must": 12},
    4: {"name": "Cleaner", "ratedpower": 1.35, "preferd": range(78, 80), "shift": range(71, 80), "must": 2},
    5: {"name": "Dryer", "ratedpower": 3.2, "preferd": range(43, 47), "shift": range(43, 60), "must": 4},
    6: {"name": "Steam", "ratedpower": 1.4, "preferd": range(80, 83), "shift": range(80, 90), "must": 3},
    7: {"name": "Fridge", "ratedpower": 0.15, "preferd": range(1, 97), "shift": range(1, 97), "must": 96},  # یخچال 24 ساعته
    8: {"name": "Lighting", "ratedpower": 0.1, "preferd": range(73, 97), "shift": range(73, 97), "must": 24},  # روشنایی 18:00 تا 02:00
    9: {"name": "Humidifier", "ratedpower": 0.2, "preferd": range(49, 73), "shift": range(49, 73), "must": 24}  # تثبیت‌کننده 12:00 تا 18:00
}
model.pflex=Param(model.A , initialize={a:appiliance[a]["ratedpower"] for a in model.A}, doc="Rated power of appiliance")
model.must=Param(model.A , initialize={a:appiliance[a]["must"] for a in model.A}, doc="Must Period in operation")
model.preferred=Set(model.A , initialize={a:appiliance[a]["preferd"] for a in model.A} , doc="Preferred period")
model.shift=Set(model.A , initialize={a:appiliance[a]["shift"] for a in model.A} , doc="Shift period")
T=range(1,97)
P_base_init={t: 0.0 for t in T}
for a , info in appiliance.items():
    p=info["ratedpower"]
    shift=list(info["shift"])
    must=info["must"]
    if len(shift)== must:
        for t in shift:
            P_base_init[t]+=p
model.P_base=Param(model.T , initialize=P_base_init, doc="Base load")
#PV Data
np.random.seed(42)
pv_product={}
for t in model.T:
    if 36<t<60:
        valuepv = np.random.normal(loc=0.5,scale=0.2)
        pv_product[t]=max(0,min(1, valuepv))
    else:
        pv_product[t]= 0
model.P_pv = Param(model.T , initialize=pv_product, doc="PV Production per day")
#Price
model.sellprice=Param(model.T , initialize=0.04 , doc="Sell price")
buy_price_schedule=[
    {"hours": range(0,8) , "price":0.1},
    {"hours": range(8,11) , "price":0.17},
    {"hours": range(11,14) , "price":0.24},
    {"hours": range(14,20) , "price":0.17},
    {"hours": range(20,23) , "price":0.24},
    {"hours": range(23,25) , "price":0.17}
]
buy_price={}
for t in model.T:
    hour = (t-1)//4
    for schedule in buy_price_schedule:
        if hour in schedule["hours"]:
            buy_price[t]=schedule["price"]
            break
model.buyprice=Param(model.T , initialize=buy_price, doc="Buy price")
#Battery Define
model.BessCapacity=Param(initialize=1,doc="Bess capacity")
model.crate=Param(initialize=1,doc="crate")
model.cfix=Param(initialize=0.2409,doc="cfix")
model.alpha=Param(initialize=0.0630,doc="alpha")
model.gamma=Param(initialize=0.0971,doc="gamma")
model.teta=Param(initialize=4.0253,doc="teta")
model.zeta=Param(initialize=1.0923,doc="zeta")
model.eta=Param(initialize=0.95, doc="Battery efficiency")
model.P_max_ch = Param(initialize=0.2 , doc="max power of Charge and Discharge")
model.P_max_net = Param(initialize=10 , doc="max power can transfer between network an home")

model.P_grid = Var(model.T , within=NonNegativeReals , doc="Power import from grid")
model.P_sell = Var(model.T ,within=NonNegativeReals , doc="Power sell to grid")
model.P_ch = Var(model.T ,within=NonNegativeReals , doc="power of Charge Battery")
model.P_dis = Var(model.T , within=NonNegativeReals , doc="power of Discharge Battery")
model.P_shift_up = Var(model.T , model.A , within=NonNegativeReals,doc="power of Shift Up")
model.P_shift_down = Var(model.T , model.A , within=NonNegativeReals , doc="power of Shift Down")
model.E = Var(model.T , within=NonNegativeReals , doc="Battery Energy")
model.R_ch = Var(model.T , within=NonNegativeReals , doc="Charge Rate")
model.R_dis = Var(model.T , within=NonNegativeReals , doc="Discharge Rate")
model.N_cycle = Var( within=NonNegativeReals , doc="Cycle Rate")
model.SOH =Var(model.T , within=NonNegativeReals , doc="State Of Charge")
model.Q_loss = Var(model.T , within=NonNegativeReals , doc="Battery Capacity Loss")
model.P_demand = Var(model.T , within=NonNegativeReals , doc="Battery Demand")

model.u_grid = Var(model.T , within=Binary , doc="sequence of  import power from grid")
model.u_sell = Var(model.T , within=Binary , doc="sequence of export power to grid")
model.u_ch = Var(model.T , within=Binary , doc="sequence of Charge Battery")
model.u_dis = Var(model.T , within=Binary , doc="sequence of Discharge Battery")
model.u_shift_up = Var(model.T , model.A , within=Binary , doc="sequence of Shift Up")
model.u_shift_down = Var(model.T , model.A , within=Binary , doc="sequence of Shift Down")

def objective(model):
    return sum((model.P_grid[t] * model.buyprice[t] - model.P_sell[t] * model.sellprice[t]) * dt for t in model.T)
model.Objective = Objective(rule=objective, sense=minimize, doc="Energy cost (buy - sell) with dt")

def maxpowerbuy(model,t):
    return model.P_grid[t] <= model.u_grid[t]*model.P_max_net
model.MaxPowerBuy=Constraint(model.T , rule=maxpowerbuy , doc="Max power buy from grid")
def maxpowersell(model,t):
    return model.P_sell[t] <= model.u_sell[t]*model.P_max_net
model.MaxPowerSell=Constraint(model.T , rule=maxpowersell , doc="Max power sell from grid")
def sequencenetwork(model,t):
    return model.u_sell[t] + model.u_grid[t] <= 1
model.SequencOfNetwork=Constraint(model.T , rule=sequencenetwork , doc="Sequenc of buy and sell")
def balancedemand(model, t):
    # LEFT: supply into the house
    supply = model.P_grid[t] + model.P_dis[t] + model.P_pv[t]
    # RIGHT: consumption in the house
    demand = model.P_base[t] \
             + sum(model.P_shift_up[t,a] - model.P_shift_down[t,a] for a in model.A) \
             + model.P_ch[t] + model.P_sell[t]
    return supply == demand
model.BalancedSupply = Constraint(model.T, rule=balancedemand, doc="Balanced Supply")

def maxcharge(model , t):
    return model.P_ch[t] <= model.P_max_ch * model.u_ch[t]
model.MaxCharge=Constraint(model.T , rule=maxcharge , doc="Max Power of Charge")
def maxdischarge(model , t):
    return model.P_dis[t] <= model.P_max_ch * model.u_dis[t]
model.MaxDischarge = Constraint(model.T , rule=maxdischarge , doc="Max Power of Discharge")
def sequencebattery(model , t):
    return model.u_ch[t] + model.u_dis[t] <= 1
model.SequenceBatery= Constraint(model.T , rule=sequencebattery , doc="Battery Sequence")
def batteryenergymax(model,t):
    return model.E[t] <= model.BessCapacity
model.BatteryEnergyMax = Constraint(model.T , rule=batteryenergymax ,doc="Battery Energy Max")
def ratecharge(model, t):
    return model.R_ch[t] == (model.eta * model.P_ch[t] * dt) / model.BessCapacity
model.RateCharge = Constraint(model.T, rule=ratecharge, doc="Charge Rate with dt")

def ratedischarge(model, t):
    return model.R_dis[t] == ((model.P_dis[t] * dt) / model.eta) / model.BessCapacity
model.RateDischarge = Constraint(model.T, rule=ratedischarge, doc="Discharge Rate with dt")

def cycleequation(model):
    return model.N_cycle == sum(model.R_dis[t] for t in model.T)
model.CycleEq = Constraint(rule=cycleequation, doc="Cycle Equation (sum of discharge rate)")


def cyleconstraints(model):
    return model.N_cycle <= 2
model.CycleCons = Constraint(rule=cyleconstraints, doc="Cycle Constraints")

def energyequation(model, t):
    if t == 1:

        return model.E[t] == 0.5 * model.BessCapacity
    else:
        return model.E[t] == model.E[t-1] + (model.eta*model.P_ch[t] - (model.P_dis[t]/model.eta)) * dt
model.EnergyBatteryEq = Constraint(model.T, rule=energyequation, doc="Energy Battery Equation with dt")


def shiftpower_up(model , t,a):
    return model.P_shift_up[t,a] == model.pflex[a]*model.u_shift_up[t,a]
model.ShiftPowerUP=Constraint(model.T , model.A , rule=shiftpower_up , doc="Shift Power Up")
def shiftpower_down(model,t,a):
    return model.P_shift_down[t,a] == model.pflex[a]*model.u_shift_down[t,a]
model.ShiftPowerDown=Constraint(model.T , model.A , rule=shiftpower_down , doc="Shift Power Down")
def shifttime_up(model, t, a):
    if t not in model.shift[a]:
        return model.u_shift_up[t,a] == 0
    return Constraint.Skip
model.ShiftTimeUp = Constraint(model.T, model.A, rule=shifttime_up, doc="Shift Time Up")

def shifttime_down(model, t, a):
    if t not in model.shift[a]:
        return model.u_shift_down[t,a] == 0
    return Constraint.Skip
model.ShiftTimeDown = Constraint(model.T, model.A, rule=shifttime_down, doc="Shift Time Down")
def no_simul_updown(model, t, a):
    return model.u_shift_up[t,a] + model.u_shift_down[t,a] <= 1
model.NoSimulUpDown = Constraint(model.T, model.A, rule=no_simul_updown, doc="No simultaneous up/down")

def demand_def(model, t):
    return model.P_demand[t] == model.P_base[t] + sum(model.P_shift_up[t,a] - model.P_shift_down[t,a] for a in model.A)
model.DemandDef = Constraint(model.T, rule=demand_def, doc="Define P_demand as base + (up-down)")

def mustrun(model, a):
    return sum(model.u_shift_up[t,a] + model.u_shift_down[t,a] for t in model.T) == model.must[a]
model.MustRun = Constraint(model.A, rule=mustrun, doc="Must Run (up+down)")
# def energy_neutral_shift(model, a):
#     return sum(model.P_shift_down[t,a]*dt for t in model.T) == sum(model.P_shift_up[t,a]*dt for t in model.T)
# model.EnergyNeutralShift = Constraint(model.A, rule=energy_neutral_shift, doc="Sum down equals sum up (energy)")


solver = SolverFactory("cbc", executable="C:\\solver\\Cbc-releases.2.10.12-w64-msvc16-md\\bin\\cbc.exe")
result = solver.solve(model, tee=True)


results = {}

for var in model.component_objects(Var, active=True):
    var_name = var.name
    var_values = {}
    for index in var:
        var_obj = var[index]
        if var_obj.is_fixed() or var_obj.value is not None:
            try:
                var_val = value(var_obj)
                var_values[index] = var_val
            except:
                continue
    if var_values:
        results[var_name] = var_values
import pandas as pd


data = {
    'Time': list(model.T),
    'P_grid': [model.P_grid[t].value for t in model.T],
    'P_sell': [model.P_sell[t].value for t in model.T],
    'P_ch': [model.P_ch[t].value for t in model.T],
    'P_dis': [model.P_dis[t].value for t in model.T],
    'P_demand': [model.P_demand[t].value for t in model.T],
    'E': [model.E[t].value for t in model.T],
    'R_ch': [model.R_ch[t].value for t in model.T],
    'R_dis': [model.R_dis[t].value for t in model.T],
    'SOH': [model.SOH[t].value for t in model.T],
    'Q_loss': [model.Q_loss[t].value for t in model.T],
    'u_grid': [model.u_grid[t].value for t in model.T],
    'u_sell': [model.u_sell[t].value for t in model.T],
    'u_ch': [model.u_ch[t].value for t in model.T],
    'u_dis': [model.u_dis[t].value for t in model.T],
    'N_cycle': [model.N_cycle.value] * len(model.T),
}


for a in model.A:
    data[f'P_shift_up_{a}'] = [model.P_shift_up[t, a].value for t in model.T]
    data[f'P_shift_down_{a}'] = [model.P_shift_down[t, a].value for t in model.T]
    data[f'u_shift_up_{a}'] = [model.u_shift_up[t, a].value for t in model.T]
    data[f'u_shift_down_{a}'] = [model.u_shift_down[t, a].value for t in model.T]


df = pd.DataFrame(data)
df.to_csv('hems_results.csv', index=False)
print("نتایج در فایل hems_results.csv ذخیره شد.")



