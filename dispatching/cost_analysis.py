import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import json
import os

class DispatchCostAnalyzer:

    def __init__(self, system_config: Dict):

        self.system_config = system_config
        self.generators = system_config['generators']
        self.storage = system_config['storage']

    def analyze_dispatch_results(self, baseline_results: Dict, proposed_results: Dict) -> Dict:

        analysis = {
            'cost_breakdown': self._analyze_cost_breakdown(baseline_results, proposed_results),
            'generation_patterns': self._analyze_generation_patterns(baseline_results, proposed_results),
            'storage_usage': self._analyze_storage_usage(baseline_results, proposed_results),
            'efficiency_metrics': self._calculate_efficiency_metrics(baseline_results, proposed_results),
            'adjustment_frequency': self._analyze_adjustment_frequency(baseline_results, proposed_results),
            'key_metrics': self._calculate_key_metrics(baseline_results, proposed_results)
        }

        return analysis

    def _analyze_cost_breakdown(self, baseline_results: Dict, proposed_results: Dict) -> Dict:

        cost_analysis = {
            'baseline': self._calculate_detailed_costs(baseline_results, 'baseline'),
            'proposed': self._calculate_detailed_costs(proposed_results, 'proposed'),
            'comparison': {}
        }

        baseline_costs = cost_analysis['baseline']
        proposed_costs = cost_analysis['proposed']

        cost_analysis['comparison'] = {
            'total_cost_change': proposed_costs['total_cost'] - baseline_costs['total_cost'],
            'total_cost_change_pct': ((proposed_costs['total_cost'] - baseline_costs['total_cost']) / baseline_costs['total_cost']) * 100,
            'generator_cost_change': proposed_costs['generator_cost'] - baseline_costs['generator_cost'],
            'storage_cost_change': proposed_costs['storage_cost'] - baseline_costs['storage_cost'],
            'carbon_cost_change': proposed_costs['carbon_cost'] - baseline_costs['carbon_cost']
        }

        return cost_analysis

    def _calculate_detailed_costs(self, results: Dict, method_name: str) -> Dict:

        total_generator_cost = 0
        total_storage_cost = 0
        total_carbon_cost = 0
        generator_costs_by_type = {}

        dispatches = results['dispatches']

        for step_idx, dispatch in enumerate(dispatches):

            for gen_id, output_series in dispatch['generator_output'].items():
                if gen_id not in generator_costs_by_type:
                    generator_costs_by_type[gen_id] = 0

                gen_data = self.generators[gen_id]
                cost_b = gen_data.get('cost_b', 20)
                cost_c = gen_data.get('cost_c', 100)

                for output in output_series:
                    step_cost = cost_b * output + cost_c
                    total_generator_cost += step_cost
                    generator_costs_by_type[gen_id] += step_cost

            for storage_id, output_series in dispatch['storage_output'].items():
                storage_data = self.storage[storage_id]
                cost_per_mwh = storage_data.get('cost_per_mwh', 5)

                for output in output_series:
                    step_cost = cost_per_mwh * output * 0.25  
                    total_storage_cost += step_cost

            carbon_penalty = self.system_config.get('constraints', {}).get('carbon_penalty', 0)
            if carbon_penalty > 0:
                for gen_id, output_series in dispatch['generator_output'].items():
                    emission_factor = self.generators[gen_id].get('emission_factor', 0)
                    for output in output_series:
                        carbon_cost = carbon_penalty * output * emission_factor
                        total_carbon_cost += carbon_cost

        return {
            'total_cost': total_generator_cost + total_storage_cost + total_carbon_cost,
            'generator_cost': total_generator_cost,
            'storage_cost': total_storage_cost,
            'carbon_cost': total_carbon_cost,
            'generator_costs_by_type': generator_costs_by_type,
            'avg_cost_per_step': (total_generator_cost + total_storage_cost + total_carbon_cost) / len(dispatches)
        }

    def _analyze_generation_patterns(self, baseline_results: Dict, proposed_results: Dict) -> Dict:

        patterns = {
            'baseline': self._extract_generation_patterns(baseline_results, 'baseline'),
            'proposed': self._extract_generation_patterns(proposed_results, 'proposed'),
            'comparison': {}
        }

        baseline_gen = patterns['baseline']
        proposed_gen = patterns['proposed']

        patterns['comparison'] = {
            'total_generation_change': proposed_gen['total_generation'] - baseline_gen['total_generation'],
            'utilization_changes': {},
            'mix_changes': {}
        }

        for gen_id in self.generators.keys():
            baseline_util = baseline_gen['utilization_rates'].get(gen_id, 0)
            proposed_util = proposed_gen['utilization_rates'].get(gen_id, 0)
            patterns['comparison']['utilization_changes'][gen_id] = proposed_util - baseline_util

            baseline_mix = baseline_gen['generation_mix'].get(gen_id, 0)
            proposed_mix = proposed_gen['generation_mix'].get(gen_id, 0)
            patterns['comparison']['mix_changes'][gen_id] = proposed_mix - baseline_mix

        return patterns

    def _extract_generation_patterns(self, results: Dict, method_name: str) -> Dict:

        total_generation_by_type = {}
        total_generation = 0
        utilization_rates = {}

        dispatches = results['dispatches']

        for dispatch in dispatches:
            for gen_id, output_series in dispatch['generator_output'].items():
                if gen_id not in total_generation_by_type:
                    total_generation_by_type[gen_id] = 0

                gen_total = sum(output_series)
                total_generation_by_type[gen_id] += gen_total
                total_generation += gen_total

        for gen_id in self.generators.keys():
            max_capacity = self.generators[gen_id]['p_max']
            total_periods = len(dispatches) * 96  
            max_possible_generation = max_capacity * total_periods
            actual_generation = total_generation_by_type.get(gen_id, 0)
            utilization_rates[gen_id] = actual_generation / max_possible_generation if max_possible_generation > 0 else 0

        generation_mix = {}
        if total_generation > 0:
            for gen_id in self.generators.keys():
                generation_mix[gen_id] = total_generation_by_type.get(gen_id, 0) / total_generation

        return {
            'total_generation': total_generation,
            'total_generation_by_type': total_generation_by_type,
            'utilization_rates': utilization_rates,
            'generation_mix': generation_mix
        }

    def _analyze_storage_usage(self, baseline_results: Dict, proposed_results: Dict) -> Dict:

        storage_analysis = {
            'baseline': self._extract_storage_usage(baseline_results),
            'proposed': self._extract_storage_usage(proposed_results),
            'comparison': {}
        }

        baseline_storage = storage_analysis['baseline']
        proposed_storage = storage_analysis['proposed']

        storage_analysis['comparison'] = {
            'total_cycles_change': proposed_storage['total_cycles'] - baseline_storage['total_cycles'],
            'avg_soc_change': proposed_storage['avg_soc'] - baseline_storage['avg_soc']
        }

        return storage_analysis

    def _extract_storage_usage(self, results: Dict) -> Dict:

        total_charging = 0
        total_discharging = 0
        total_cycles = 0
        avg_soc = 0

        dispatches = results['dispatches']

        for dispatch in dispatches:
            for storage_id, output_series in dispatch['storage_output'].items():
                for output in output_series:
                    if output > 0:  
                        total_discharging += output
                    else:  
                        total_charging += abs(output)

            for storage_id, soc_series in dispatch['storage_soc'].items():
                avg_soc += np.mean(soc_series)

        total_cycles = min(total_charging, total_discharging)
        avg_soc = avg_soc / len(dispatches) if dispatches else 0

        return {
            'total_charging': total_charging,
            'total_discharging': total_discharging,
            'total_cycles': total_cycles,
            'avg_soc': avg_soc
        }

    def _calculate_efficiency_metrics(self, baseline_results: Dict, proposed_results: Dict) -> Dict:

        baseline_efficiency = self._calculate_single_efficiency(baseline_results)
        proposed_efficiency = self._calculate_single_efficiency(proposed_results)

        return {
            'baseline': baseline_efficiency,
            'proposed': proposed_efficiency,
            'efficiency_change': proposed_efficiency - baseline_efficiency
        }

    def _calculate_single_efficiency(self, results: Dict) -> float:

        total_cost = sum(results['costs'])
        total_generation = 0

        for dispatch in results['dispatches']:
            for gen_id, output_series in dispatch['generator_output'].items():
                total_generation += sum(output_series)

        return total_generation / total_cost if total_cost > 0 else 0

    def _analyze_adjustment_frequency(self, baseline_results: Dict, proposed_results: Dict) -> Dict:

        baseline_freq = self._calculate_adjustment_frequency(baseline_results)
        proposed_freq = self._calculate_adjustment_frequency(proposed_results)

        return {
            'baseline': baseline_freq,
            'proposed': proposed_freq,
            'frequency_change': proposed_freq - baseline_freq
        }

    def _calculate_adjustment_frequency(self, results: Dict) -> float:

        total_adjustments = 0
        total_possible_adjustments = 0

        dispatches = results['dispatches']

        for step_idx in range(1, len(dispatches)):
            current_dispatch = dispatches[step_idx]
            previous_dispatch = dispatches[step_idx - 1]

            for gen_id in self.generators.keys():
                current_output = current_dispatch['generator_output'].get(gen_id, [0])
                previous_output = previous_dispatch['generator_output'].get(gen_id, [0])

                if len(current_output) > 0 and len(previous_output) > 0:

                    if abs(current_output[0] - previous_output[0]) > 50:  
                        total_adjustments += 1
                    total_possible_adjustments += 1

        return total_adjustments / total_possible_adjustments if total_possible_adjustments > 0 else 0

    def _calculate_key_metrics(self, baseline_results: Dict, proposed_results: Dict) -> Dict:

        baseline_metrics = self._calculate_single_key_metrics(baseline_results, 'baseline')
        proposed_metrics = self._calculate_single_key_metrics(proposed_results, 'proposed')

        improvements = {}
        for metric in baseline_metrics.keys():
            if baseline_metrics[metric] != 0:
                if metric in ['carbon_emission_intensity', 'unit_adjustment_frequency', 'additional_energy_consumption_rate']:

                    improvements[metric] = ((baseline_metrics[metric] - proposed_metrics[metric]) / baseline_metrics[metric]) * 100
                else:

                    improvements[metric] = ((baseline_metrics[metric] - proposed_metrics[metric]) / baseline_metrics[metric]) * 100
            else:
                improvements[metric] = 0

        return {
            'baseline': baseline_metrics,
            'proposed': proposed_metrics,
            'improvements': improvements
        }

    def _calculate_single_key_metrics(self, results: Dict, method_name: str) -> Dict:

        dispatches = results['dispatches']
        total_steps = len(dispatches)

        total_cost = sum(results['costs'])

        total_days = (total_steps * 1) / 96  
        avg_daily_operating_cost = total_cost / total_days if total_days > 0 else 0

        total_generation = 0
        total_carbon_emission = 0

        for dispatch in dispatches:
            for gen_id, output_series in dispatch['generator_output'].items():
                emission_factor = self.generators[gen_id].get('emission_factor', 0)  
                for output in output_series:

                    generation_mwh = output * 0.25  
                    total_generation += generation_mwh

                    carbon_emission = output * emission_factor * 0.25
                    total_carbon_emission += carbon_emission

        carbon_emission_intensity = total_carbon_emission / total_generation if total_generation > 0 else 0

        total_adjustments = 0
        total_capacity_hours = 0

        for step_idx in range(1, len(dispatches)):
            current_dispatch = dispatches[step_idx]
            previous_dispatch = dispatches[step_idx - 1]

            for gen_id in self.generators.keys():
                current_output = current_dispatch['generator_output'].get(gen_id, [])
                previous_output = previous_dispatch['generator_output'].get(gen_id, [])

                if len(current_output) > 0 and len(previous_output) > 0:

                    output_change = abs(current_output[0] - previous_output[0])
                    if output_change > 50:  
                        total_adjustments += 1

                    max_capacity = self.generators[gen_id]['p_max']
                    capacity_hours = max_capacity * 0.25  
                    total_capacity_hours += capacity_hours

        unit_adjustment_frequency = total_adjustments / total_capacity_hours if total_capacity_hours > 0 else 0

        total_storage_loss = 0

        for dispatch in dispatches:
            for storage_id, output_series in dispatch['storage_output'].items():
                efficiency = self.storage[storage_id].get('efficiency', 0.95)
                for output in output_series:
                    if output < 0:  

                        charging_loss = abs(output) * (1 - efficiency) * 0.25
                        total_storage_loss += charging_loss

        renewable_rate = self._calculate_renewable_rate(dispatches)

        return {
            'avg_daily_operating_cost': avg_daily_operating_cost,  
            'carbon_emission_intensity': carbon_emission_intensity,  
            'unit_adjustment_frequency': unit_adjustment_frequency,  
            'additional_energy_consumption_rate': renewable_rate  
        }

    def _calculate_renewable_rate(self, dispatches: List[Dict]) -> float:

        total_generation = 0.0  
        renewable_generation = 0.0  

        for dispatch in dispatches:
            for gen_id, output_series in dispatch['generator_output'].items():

                gen_output = sum(output * 0.25 for output in output_series)  
                total_generation += gen_output

                if 'wind' in gen_id.lower() or 'solar' in gen_id.lower():
                    renewable_generation += gen_output

        if total_generation == 0:
            return 0.0

        renewable_rate = (renewable_generation / total_generation) * 100.0

        return max(0.0, min(100.0, renewable_rate))

    def print_detailed_analysis(self, analysis: Dict):

        print("\n" + "=" * 80)
        print("详细成本分析报告")
        print("=" * 80)

        if 'key_metrics' in analysis:
            key_metrics = analysis['key_metrics']
            baseline_key = key_metrics['baseline']
            proposed_key = key_metrics['proposed']
            improvements = key_metrics['improvements']

            print("\n 调度指标对比:")
            print("=" * 50)

            print(f"1. 平均每日运营成本:")
            print(f"   基线方法: {baseline_key['avg_daily_operating_cost']:,.2f} ¥/天")
            print(f"   提出方法: {proposed_key['avg_daily_operating_cost']:,.2f} ¥/天")
            if improvements['avg_daily_operating_cost'] > 0:
                print(f"   改进效果:  降低 {improvements['avg_daily_operating_cost']:.2f}%")
            else:
                print(f"   改进效果:  增加 {abs(improvements['avg_daily_operating_cost']):.2f}%")

            print(f"\n2. 碳排放强度:")
            print(f"   基线方法: {baseline_key['carbon_emission_intensity']:.4f} tCO2/MWh")
            print(f"   提出方法: {proposed_key['carbon_emission_intensity']:.4f} tCO2/MWh")
            if improvements['carbon_emission_intensity'] > 0:
                print(f"   改进效果:  降低 {improvements['carbon_emission_intensity']:.2f}%")
            else:
                print(f"   改进效果:  增加 {abs(improvements['carbon_emission_intensity']):.2f}%")

            print(f"\n3. 单位调整次数:")
            print(f"   基线方法: {baseline_key['unit_adjustment_frequency']:.6f} 次/(MW·h)")
            print(f"   提出方法: {proposed_key['unit_adjustment_frequency']:.6f} 次/(MW·h)")
            if improvements['unit_adjustment_frequency'] > 0:
                print(f"   改进效果:  降低 {improvements['unit_adjustment_frequency']:.2f}%")
            else:
                print(f"   改进效果:  增加 {abs(improvements['unit_adjustment_frequency']):.2f}%")

            print(f"\n4. 新能源使用率:")
            print(f"   基线方法: {baseline_key['additional_energy_consumption_rate']:.2f}%")
            print(f"   提出方法: {proposed_key['additional_energy_consumption_rate']:.2f}%")
            if improvements['additional_energy_consumption_rate'] < 0:
                print(f"   改进效果:  提升 {abs(improvements['additional_energy_consumption_rate']):.2f}%")
            else:
                print(f"   改进效果:  降低 {abs(improvements['additional_energy_consumption_rate']):.2f}%")
            '''
            # 总体评估
            positive_improvements = sum(1 for imp in improvements.values() if imp > 0)
            total_metrics = len(improvements)

            print(f"\n 总体评估:")
            print(f"   改进指标数: {positive_improvements}/{total_metrics}")
            if positive_improvements >= total_metrics * 0.75:
                print("   总体评价:  显著改进")
            elif positive_improvements >= total_metrics * 0.5:
                print("   总体评价:   部分改进")
            else:
                print("   总体评价:  需要优化")
        # 成本分解
        cost_breakdown = analysis['cost_breakdown']
        print("\n" + "=" * 50)
        print("详细成本构成分析:")
        print("=" * 50)

        baseline_costs = cost_breakdown['baseline']
        proposed_costs = cost_breakdown['proposed']
        comparison = cost_breakdown['comparison']

        print(f"基线方法总成本: {baseline_costs['total_cost']:,.2f} ¥")
        print(f"  - 机组成本: {baseline_costs['generator_cost']:,.2f} ¥ ({baseline_costs['generator_cost']/baseline_costs['total_cost']*100:.1f}%)")
        print(f"  - 储能成本: {baseline_costs['storage_cost']:,.2f} ¥ ({baseline_costs['storage_cost']/baseline_costs['total_cost']*100:.1f}%)")
        print(f"  - 碳排放成本: {baseline_costs['carbon_cost']:,.2f} ¥ ({baseline_costs['carbon_cost']/baseline_costs['total_cost']*100:.1f}%)")

        print(f"\n提出方法总成本: {proposed_costs['total_cost']:,.2f} ¥")
        print(f"  - 机组成本: {proposed_costs['generator_cost']:,.2f} ¥ ({proposed_costs['generator_cost']/proposed_costs['total_cost']*100:.1f}%)")
        print(f"  - 储能成本: {proposed_costs['storage_cost']:,.2f} ¥ ({proposed_costs['storage_cost']/proposed_costs['total_cost']*100:.1f}%)")
        print(f"  - 碳排放成本: {proposed_costs['carbon_cost']:,.2f} ¥ ({proposed_costs['carbon_cost']/proposed_costs['total_cost']*100:.1f}%)")

        print(f"\n成本变化:")
        print(f"  - 总成本变化: {comparison['total_cost_change']:+,.2f} ¥ ({comparison['total_cost_change_pct']:+.2f}%)")
        print(f"  - 机组成本变化: {comparison['generator_cost_change']:+,.2f} ¥")
        print(f"  - 储能成本变化: {comparison['storage_cost_change']:+,.2f} ¥")
        print(f"  - 碳排放成本变化: {comparison['carbon_cost_change']:+,.2f} ¥")

        # 发电模式分析
        generation_patterns = analysis['generation_patterns']
        print("\n2. 发电模式分析:")
        print("-" * 40)

        baseline_gen = generation_patterns['baseline']
        proposed_gen = generation_patterns['proposed']
        gen_comparison = generation_patterns['comparison']

        print("机组利用率比较:")
        for gen_id in self.generators.keys():
            baseline_util = baseline_gen['utilization_rates'].get(gen_id, 0)
            proposed_util = proposed_gen['utilization_rates'].get(gen_id, 0)
            change = gen_comparison['utilization_changes'].get(gen_id, 0)
            print(f"  {gen_id}: {baseline_util:.1%} → {proposed_util:.1%} ({change:+.1%})")

        print("\n发电结构变化:")
        for gen_id in self.generators.keys():
            baseline_mix = baseline_gen['generation_mix'].get(gen_id, 0)
            proposed_mix = proposed_gen['generation_mix'].get(gen_id, 0)
            change = gen_comparison['mix_changes'].get(gen_id, 0)
            print(f"  {gen_id}: {baseline_mix:.1%} → {proposed_mix:.1%} ({change:+.1%})")

        # 调整频率分析
        adjustment_freq = analysis['adjustment_frequency']
        print("\n3. 调整频率分析:")
        print("-" * 40)
        print(f"基线方法调整频率: {adjustment_freq['baseline']:.1%}")
        print(f"提出方法调整频率: {adjustment_freq['proposed']:.1%}")
        print(f"频率变化: {adjustment_freq['frequency_change']:+.1%}")

        # 效率分析
        efficiency = analysis['efficiency_metrics']
        print("\n4. 效率分析:")
        print("-" * 40)
        print(f"基线方法效率: {efficiency['baseline']:.4f} MW/¥")
        print(f"提出方法效率: {efficiency['proposed']:.4f} MW/¥")
        

        print(f"效率变化: {efficiency['efficiency_change']:+.4f} MW/¥ ({efficiency['efficiency_change']/efficiency['baseline']*100:+.2f}%)")
        '''
    def calculate_energy_efficiency(self):

        if not self.storage_schedule:
            return 0.0

        total_charge = 0.0  
        total_discharge = 0.0  

        for step_data in self.storage_schedule:
            battery_power = step_data.get('battery_power', 0.0)
            if battery_power > 0:  
                total_charge += battery_power / 4.0  
            elif battery_power < 0:  
                total_discharge += abs(battery_power) / 4.0

        if total_charge == 0:
            return 100.0

        efficiency = (total_discharge / total_charge) * 100.0

        efficiency = max(75.0, min(95.0, efficiency))

        return efficiency

def analyze_experiment_results(baseline_results: Dict, proposed_results: Dict, system_config: Dict):

    analyzer = DispatchCostAnalyzer(system_config)
    analysis = analyzer.analyze_dispatch_results(baseline_results, proposed_results)
    analyzer.print_detailed_analysis(analysis)

    return analysis