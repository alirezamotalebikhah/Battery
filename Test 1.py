

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pyomo.environ import *
from torch.fx.experimental.migrate_gradual_types.constraint import Constraint

#define model
model = ConcreteModel(doc="Home energy Management")
#define Set
model.T=RangeSet(1,96,doc="Time Period")
model.A=RangeSet(1,6,doc="Appiliance")

#Define Parameters
#Appilince Data
appiliance={
    1 : {"name":"Dishwasher" , "ratedpower":2.8 , "preferd":range(81,87) , "shift":range(61,87) , "must":6},
    2 : {"name":"Washing" , "ratedpower":3.5 , "preferd":range(40,43) , "shift":range(37,45) , "must":3},
    3 : {"name":"Pump" , "ratedpower":1.5 , "preferd":range(41,53) , "shift":range(25,53) , "must":12},
    4 : {"name":"Cleaner" , "ratedpower":1.35 , "preferd":range(78,80) , "shift":range(71,80) , "must":2},
    5 : {"name":"Dryer" , "ratedpower":3.2 , "preferd":range(43,47) , "shift":range(43,60) , "must":4},
    6 : {"name":"Steam" , "ratedpower":1.4 , "preferd":range(80,83) , "shift":range(80,90) , "must":3}
}
model.pflex=Param(model.A , initialize={a:appiliance[a]["ratedpower"] for a in model.A}, doc="Rated power of appiliance")
model.must=Param(model.A , initialize={a:appiliance[a]["must"] for a in model.A}, doc="Must Period in operation")
model.preferred=Set(model.A , initialize={a:appiliance[a]["preferd"] for a in model.A} , doc="Preferred period")
model.shift=Set(model.A , initialize={a:appiliance[a]["shift"] for a in model.A} , doc="Shift period")
#PV Data
np.random.seed(42)
pv_product={}
for t in model.T:
    if 36<t<60:
        value = np.random.normal(loc=0.5,scale=0.2)
        pv_product[t]=max(0,min(1, value))
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
model.BessCapacity=Param(initialize=10,doc="Bess capacity")
model.crate=Param(initialize=1,doc="crate")
model.cfix=Param(initialize=0.2409,doc="cfix")
model.alpha=Param(initialize=0.0630,doc="alpha")
model.gamma=Param(initialize=0.0971,doc="gamma")
model.teta=Param(initialize=4.0253,doc="teta")
model.zeta=Param(initialize=1.0923,doc="zeta")
model.eta=Param(initialize=0.95, doc="Battery efficiency")
model.P_max_ch = Param(initialize=2 , doc="max power of Charge and Discharge")

model.P_grid = Var(model.T , bounds=(0,100) , within=NonNegativeReals , doc="Power import from grid")
model.P_sell = Var(model.T , bounds=(0,100) , within=NonNegativeReals , doc="Power sell to grid")
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
    return sum(model.P_grid[t] - model.P_sell[t] for t in model.T)
model.Objective = Objective(rule=objective, sense=minimize , doc="Objective function")
def balancedemand(model , t):
    demand =model.P_demand[t] + model.P_ch[t] + model.P_sell[t] + sum(model.P_shift_up[t,a] for a in model.A)
    generation = model.P_grid[t] + model.P_dis[t] + sum(model.P_shift_down[t,a] for a in model.A) +model.P_pv[t]
    return demand == generation
model.BalancedSupply = Constraint(model.T , rule=balancedemand , doc="Balanced Supply")
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
def ratecharge(model , t):
    return model.R_ch[t] == (model.eta * model.P_ch[t])/model.BessCapacity
model.RateCharge = Constraint(model.T , rule=ratecharge , doc="Charge Rate")
def ratedischarge(model , t):
    return model.R_dis[t] == (model.P_dis[t])/(model.eta * model.BessCapacity)
model.RateDischarge=Constraint(model.T , rule=ratedischarge , doc="Discharge Rate")
def cycleequation(model):
    return model.N_cycle == sum(model.R_dis[t] for t in model.T)
model.CycleEq=Constraint(rule=cycleequation , doc="Cycle Equation")
def cyleconstraints(model):
    return model.N_cycle <= 0.5
model.CycleCons=Constraint(rule=cyleconstraints , doc="Cycle Constraints")
def energyequation(model , t):
    if t==1 :
        return model.E[t] == 2
    else:
        return model.E[t] == model.E[t-1] + model.eta * model.P_ch[t] - (model.P_dis[t]/model.eta)
model.EnergyBatteryEq=Constraint(model.T , rule=energyequation , doc="Energy Battery Equation")


