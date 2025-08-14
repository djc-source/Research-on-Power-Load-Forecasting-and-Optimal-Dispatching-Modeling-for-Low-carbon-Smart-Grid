import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
import warnings

class OptimalDispatcher:

    def __init__(self, config: Dict):

        self.config = config
        self.generators = config.get('generators', {})
        self.storage = config.get('storage', {})
        self.constraints = config.get('constraints', {})
        self.solver_config = config.get('solver', {'name': 'glpk'})

        self._validate_config()

    def _validate_config(self):

        required_keys = ['generators', 'storage', 'constraints']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"配置中缺少必需的键: {key}")

    def create_model(self, load_forecast: np.ndarray, time_periods: int = 96) -> pyo.ConcreteModel:

        model = pyo.ConcreteModel()

        model.T = pyo.RangeSet(1, time_periods)  
        model.G = pyo.Set(initialize=list(self.generators.keys()))  
        model.S = pyo.Set(initialize=list(self.storage.keys()))     

        model.demand = pyo.Param(model.T, initialize=dict(enumerate(load_forecast, 1)))

        model.P_g = pyo.Var(model.G, model.T, 
                           domain=pyo.NonNegativeReals,
                           doc="机组出力 [MW]")

        model.E_s = pyo.Var(model.S, model.T, 
                           domain=pyo.Reals,
                           doc="储能充放电功率 [MW]")

        model.SOC = pyo.Var(model.S, model.T,
                           domain=pyo.NonNegativeReals,
                           doc="储能荷电状态 [MWh]")

        self._add_constraints(model)

        self._add_objective(model)

        return model

    def _add_constraints(self, model: pyo.ConcreteModel):

        def supply_demand_balance_rule(model, t):

            return (sum(model.P_g[g, t] for g in model.G) + 
                   sum(model.E_s[s, t] for s in model.S) == 
                   model.demand[t])

        model.supply_demand_balance = pyo.Constraint(
            model.T, rule=supply_demand_balance_rule,
            doc="供需平衡约束"
        )

        def generator_min_output_rule(model, g, t):

            gen_data = self.generators[g]
            return model.P_g[g, t] >= gen_data['p_min']

        def generator_max_output_rule(model, g, t):

            gen_data = self.generators[g]
            return model.P_g[g, t] <= gen_data['p_max']

        model.generator_min_output = pyo.Constraint(
            model.G, model.T, 
            rule=generator_min_output_rule,
            doc="机组最小出力约束"
        )

        model.generator_max_output = pyo.Constraint(
            model.G, model.T, 
            rule=generator_max_output_rule,
            doc="机组最大出力约束"
        )

        def ramp_up_constraint_rule(model, g, t):

            if t == 1:
                return pyo.Constraint.Skip
            gen_data = self.generators[g]
            ramp_up = gen_data.get('ramp_up', float('inf'))
            return model.P_g[g, t] - model.P_g[g, t-1] <= ramp_up

        def ramp_down_constraint_rule(model, g, t):

            if t == 1:
                return pyo.Constraint.Skip
            gen_data = self.generators[g]
            ramp_down = gen_data.get('ramp_down', float('inf'))
            return model.P_g[g, t-1] - model.P_g[g, t] <= ramp_down

        model.ramp_up_constraint = pyo.Constraint(
            model.G, model.T, rule=ramp_up_constraint_rule,
            doc="机组向上爬坡约束"
        )

        model.ramp_down_constraint = pyo.Constraint(
            model.G, model.T, rule=ramp_down_constraint_rule,
            doc="机组向下爬坡约束"
        )

        def storage_soc_bounds_rule(model, s, t):

            storage_data = self.storage[s]
            return (storage_data['soc_min'], 
                   model.SOC[s, t], 
                   storage_data['soc_max'])

        model.storage_soc_bounds = pyo.Constraint(
            model.S, model.T, 
            rule=storage_soc_bounds_rule,
            doc="储能SOC约束"
        )

        def storage_soc_dynamic_rule(model, s, t):

            if t == 1:

                storage_data = self.storage[s]
                initial_soc = storage_data.get('initial_soc', storage_data['soc_max'] * 0.5)
                return model.SOC[s, t] == initial_soc - model.E_s[s, t] * 0.25  
            else:

                efficiency = self.storage[s].get('efficiency', 0.95)
                dt = 0.25  

                return model.SOC[s, t] == model.SOC[s, t-1] - model.E_s[s, t] * dt / efficiency

        model.storage_soc_dynamic = pyo.Constraint(
            model.S, model.T, 
            rule=storage_soc_dynamic_rule,
            doc="储能SOC动态约束"
        )

        def storage_power_bounds_rule(model, s, t):

            storage_data = self.storage[s]
            return (-storage_data['p_max'], 
                   model.E_s[s, t], 
                   storage_data['p_max'])

        model.storage_power_bounds = pyo.Constraint(
            model.S, model.T, 
            rule=storage_power_bounds_rule,
            doc="储能功率约束"
        )

        if 'carbon_limit' in self.constraints:
            def carbon_emission_rule(model, t):

                total_emission = sum(
                    model.P_g[g, t] * self.generators[g].get('emission_factor', 0)
                    for g in model.G
                )
                return total_emission <= self.constraints['carbon_limit']

            model.carbon_emission = pyo.Constraint(
                model.T, rule=carbon_emission_rule,
                doc="碳排放约束"
            )

    def _add_objective(self, model: pyo.ConcreteModel):

        def objective_rule(model):

            generator_cost = sum(
                self._generator_cost(model.P_g[g, t], g) 
                for g in model.G for t in model.T
            )

            storage_cost = sum(
                self._storage_cost(model.E_s[s, t], s)
                for s in model.S for t in model.T
            )

            emission_penalty = 0
            if 'carbon_penalty' in self.constraints:
                lambda_carbon = self.constraints['carbon_penalty']
                emission_penalty = sum(
                    lambda_carbon * model.P_g[g, t] * 
                    self.generators[g].get('emission_factor', 0)
                    for g in model.G for t in model.T
                )

            return generator_cost + storage_cost + emission_penalty

        model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    def _generator_cost(self, power: pyo.Var, gen_id: str) -> pyo.Expression:

        gen_data = self.generators[gen_id]

        b = gen_data.get('cost_b', 20)    
        c = gen_data.get('cost_c', 100)   

        return b * power + c

    def _storage_cost(self, power: pyo.Var, storage_id: str) -> pyo.Expression:

        storage_data = self.storage[storage_id]
        cost_per_mwh = storage_data.get('cost_per_mwh', 5)  

        return cost_per_mwh * power * 0.25  

    def solve(self, model: pyo.ConcreteModel, 
              compensation: Optional[np.ndarray] = None) -> Dict:

        if compensation is not None:

            demand_dict = {}
            for t in model.T:
                original_demand = pyo.value(model.demand[t])
                compensated_demand = original_demand + compensation[t-1]
                demand_dict[t] = compensated_demand

            model.del_component(model.demand)
            model.demand = pyo.Param(model.T, initialize=demand_dict)

        solver_name = self.solver_config.get('name', 'glpk')
        try:
            solver = pyo.SolverFactory(solver_name)
        except Exception:
            print(f"警告：无法找到求解器 {solver_name}，尝试使用 glpk")
            solver = pyo.SolverFactory('glpk')

        solver_options = self.solver_config.get('options', {})
        for key, value in solver_options.items():
            solver.options[key] = value

        try:
            results = solver.solve(model, tee=False)

            if (results.solver.status == SolverStatus.ok and 
                results.solver.termination_condition == TerminationCondition.optimal):

                return self._extract_solution(model, results)
            else:
                raise RuntimeError(f"求解失败: {results.solver.termination_condition}")

        except Exception as e:
            raise RuntimeError(f"求解过程中出现错误: {e}")

    def _extract_solution(self, model: pyo.ConcreteModel, results) -> Dict:

        solution = {
            'status': 'optimal',
            'objective_value': pyo.value(model.obj),
            'generator_output': {},
            'storage_output': {},
            'storage_soc': {},
            'total_cost': pyo.value(model.obj),
            'supply_demand': []
        }

        for g in model.G:
            solution['generator_output'][g] = [
                pyo.value(model.P_g[g, t]) for t in model.T
            ]

        for s in model.S:
            solution['storage_output'][s] = [
                pyo.value(model.E_s[s, t]) for t in model.T
            ]
            solution['storage_soc'][s] = [
                pyo.value(model.SOC[s, t]) for t in model.T
            ]

        for t in model.T:
            supply = sum(pyo.value(model.P_g[g, t]) for g in model.G)
            storage = sum(pyo.value(model.E_s[s, t]) for s in model.S)
            demand = pyo.value(model.demand[t])
            solution['supply_demand'].append({
                'time': t,
                'supply': supply,
                'storage': storage,
                'demand': demand,
                'balance': supply + storage - demand
            })

        return solution

    def calculate_metrics(self, solution: Dict, actual_load: Optional[np.ndarray] = None) -> Dict:

        metrics = {
            'total_cost': solution['total_cost'],
            'avg_cost_per_mwh': 0,
            'carbon_emission': 0,
            'renewable_utilization': 0,
            'generator_adjustments': 0,
            'max_prediction_error': 0
        }

        total_generation = 0
        for g, output in solution['generator_output'].items():
            total_generation += sum(output)

        if total_generation > 0:
            metrics['avg_cost_per_mwh'] = solution['total_cost'] / total_generation

        for g, output in solution['generator_output'].items():
            emission_factor = self.generators[g].get('emission_factor', 0)
            metrics['carbon_emission'] += sum(p * emission_factor for p in output)

        total_adjustments = 0
        for g, output in solution['generator_output'].items():
            for i in range(1, len(output)):
                if abs(output[i] - output[i-1]) > 1.0:  
                    total_adjustments += 1
        metrics['generator_adjustments'] = total_adjustments

        if actual_load is not None:
            forecast_load = [item['demand'] for item in solution['supply_demand']]
            errors = [abs(actual_load[i] - forecast_load[i]) 
                     for i in range(min(len(actual_load), len(forecast_load)))]
            if errors:
                metrics['max_prediction_error'] = max(errors)

        return metrics