import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

print('Analysis module loaded')


class InterdigitationAnalysis:
    def __init__(self, top_file, traj_file, lipids, NL, water):
        self.u = mda.Universe(top_file, traj_file)
        self.lipids = lipids
        self.NL = NL
        self.water = water
        
    # get histogram densities for the overlap
    def density_frame(self, pos, mass, pbc, bins):
        dz = bins[1] - bins[0]
        h, _ = np.histogram(pos, weights=mass, bins=bins)
        h /= pbc[0] * pbc[1] * dz * 0.602214
        return h
    
    # calculating overlap of lipids
    def cal_overlap(self, d1, d2):
        thr = 0.1
        d_sum = d1 + d2
        d_mul = d1 * d2
        d_sum[d1 + d2 < thr] = 1
        d_mul[d1 + d2 < thr] = 0
        ov = 4 * d_mul / d_sum**2
        return ov
    
    # Interdigitation using integral over z-dim
    def cal_inter(self, ov, dz):
        interdigitation = np.sum(ov) * dz
        return interdigitation / 10  # to nm
    

    def setup_atom_groups(self):
        halfz = self.u.dimensions[2] / 2
        C2 = ' '.join([f'C2{i}' for i in range(2, 22)])
        C3 = ' '.join([f'C3{i}' for i in range(2, 22)])
        tail_atoms = ' '.join(C2.split() + C3.split())

        # lipid_selection = ' or '.join([f'resname {lipid}' for lipid in self.lipids])
        # upper_sel = f'(same residue as ({lipid_selection}) and name P and prop z>{halfz}) and (name {tail_atoms})'
        # lower_sel = f'(same residue as ({lipid_selection}) and name P and prop z<{halfz}) and (name {tail_atoms})'

        lipid_selection = ' or '.join([f'resname {lipid}' for lipid in self.lipids])
        us = f'(same residue as ({lipid_selection}) and name P and prop z>{halfz}) and (name {" ".join(tail_atoms.split())})'
        ls = f'(same residue as ({lipid_selection}) and name P and prop z<{halfz}) and (name {" ".join(tail_atoms.split())})'

        groups = {
            'memb': self.u.select_atoms(lipid_selection),
            'umemb': self.u.select_atoms(us),
            'lmemb': self.u.select_atoms(ls),
            'trio': self.u.select_atoms(f'resname {self.NL}'),
            'water': self.u.select_atoms(f'resname {self.water}')
        }
        return groups

    def calculate_densities(self, groups, pbc, bins):
        d0 = self.density_frame(groups['memb'].positions[:, 2], groups['memb'].masses, pbc=pbc, bins=bins)
        d1 = self.density_frame(groups['trio'].positions[:, 2], groups['trio'].masses, pbc=pbc, bins=bins)
        d_water = self.density_frame(groups['water'].positions[:, 2], groups['water'].masses, pbc=pbc, bins=bins)
        return d0, d1, d_water

    def calculate_overlap_and_inter(self, d0, d1, dz):
        tov = self.cal_overlap(d0, d1)
        tin = self.cal_inter(tov, dz)
        return tov, tin

    def calculate_strong_resids(self, trio_pos, utz, ltz, names, resids):
        boolArray = ((trio_pos[:, 2] > utz) | (trio_pos[:, 2] < ltz)) & np.char.startswith(names, 'O')
        strong_resids = resids[boolArray]
        r = np.unique(strong_resids, return_counts=True)
        boolArray = (r[1] == 6)
        strong_resids = r[0][boolArray]
        return strong_resids

    def calculate_densities_for_resids(self, trio_pos, resids, strong_resids, groups, pbc, bins):
        boolArray = np.isin(resids, strong_resids)
        pp = trio_pos[boolArray]
        mm = groups['trio'].masses[boolArray]
        d2 = self.density_frame(pp[:, 2], mm, pbc, bins)
        return d2

    def calculate_overlap_and_inter_for_resids(self, d0, d2, dz):
        sov = self.cal_overlap(d0, d2)
        sin = self.cal_inter(sov, dz)
        return sov, sin

    def calculate_densities_for_inverted_resids(self, trio_pos, resids, strong_resids, groups, pbc, bins):
        boolArray = np.isin(resids, strong_resids, invert=True)
        pp = trio_pos[boolArray]
        mm = groups['trio'].masses[boolArray]
        d3 = self.density_frame(pp[:, 2], mm, pbc, bins)
        return d3

    def calculate_overlap_and_inter_for_inverted_resids(self, d0, d3, dz):
        wov = self.cal_overlap(d0, d3)
        win = self.cal_inter(wov, dz)
        return wov, win

    def interdigit(self, nbins=50, nblocks=5, b=0, e=None):
        times, zs, total_inter, strong_inter, weak_inter, d0_densities, d1_densities, d2_densities, d3_densities, total_ov, strong_ov, weak_ov, strong_num = ([] for _ in range(13))
        d0_series, d1_series, d2_series, d3_series, d_water_series = ([] for _ in range(5))
        groups = self.setup_atom_groups()
        names = groups['trio'].names.astype(str)
        resids = groups['trio'].resids
        numP = self.u.select_atoms('name P').n_atoms
        for ts in self.u.trajectory[b:e]:
            if int(ts.time / 1000) % 1000 == 0:
                print("analyzing %d us.... " % (ts.time / 1000 / 1000))
            pbc = self.u.dimensions
            bins = np.linspace(0, pbc[2], nbins + 1)
            dz = bins[1] - bins[0]
            trio_pos = groups['trio'].positions
            utz = np.average(groups['umemb'].positions[:, 2])
            ltz = np.average(groups['lmemb'].positions[:, 2])
            d0, d1, d_water = self.calculate_densities(groups, pbc, bins)
            d0_densities.append(d0)
            d1_densities.append(d1)
            d_water_densities = [d_water]
            d0_series.append(d0)
            d1_series.append(d1)
            d_water_series.append(d_water)
            tov, tin = self.calculate_overlap_and_inter(d0, d1, dz)
            total_ov.append(tov)
            total_inter.append(tin)
            strong_resids = self.calculate_strong_resids(trio_pos, utz, ltz, names, resids)
            d2 = self.calculate_densities_for_resids(trio_pos, resids, strong_resids, groups, pbc, bins)
            d2_densities.append(d2)
            d2_series.append(d2)
            sov, sin = self.calculate_overlap_and_inter_for_resids(d0, d2, dz)
            strong_ov.append(sov)
            strong_inter.append(sin)
            d3 = self.calculate_densities_for_inverted_resids(trio_pos, resids, strong_resids, groups, pbc, bins)
            d3_densities.append(d3)
            d3_series.append(d3)
            wov, win = self.calculate_overlap_and_inter_for_inverted_resids(d0, d3, dz)
            weak_ov.append(wov)
            weak_inter.append(win)
            strong_num.append(len(strong_resids))
            times.append(ts.time / 1000)
            zs.append(pbc[2] / 10)
        strong_num = np.array(strong_num)
        XX = np.linspace(0, np.average(zs), nbins)
        results = {}

        results['inter'] = {}
        results['inter']['total'] = np.transpose([times, total_inter])
        results['inter']['strong'] = np.transpose([times, strong_inter])
        results['inter']['weak'] = np.transpose([times, weak_inter])

        results['ov'] = {}
        results['ov']['total'] = np.transpose([XX, np.average(total_ov, axis=0)])
        results['ov']['strong'] = np.transpose([XX, np.average(strong_ov, axis=0)])
        results['ov']['weak'] = np.transpose([XX, np.average(weak_ov, axis=0)])

        results['ratio'] = {}
        results['ratio']['num'] = np.transpose([times, strong_num])
        results['ratio']['trio-to-pl'] = np.transpose([times, strong_num / numP])
        results['ratio']['trio-to-pl+trio'] = np.transpose([times, strong_num / (numP + strong_num)])

        results['density'] = {
        'PL': np.transpose([XX, np.average(d0_densities, axis=0)]),
        'TRIO': np.transpose([XX, np.average(d1_densities, axis=0)]),
        'SURF-TRIO': np.transpose([XX, np.average(d2_densities, axis=0)]),
        'CORE-TRIO': np.transpose([XX, np.average(d3_densities, axis=0)]),
        'water': np.transpose([XX, np.average(d_water_densities, axis=0)]),
        'PL_series': np.array(d0_series),
        'TRIO_series': np.array(d1_series),
        'SURF-TRIO_series': np.array(d2_series),
        'CORE-TRIO_series': np.array(d3_series),
        'water_series': np.array(d_water_series)
        }


        # results['density'] = {}
        # results['density']['PL'] = np.transpose([XX, np.average(d0_densities, axis=0)])
        # results['density']['TRIO'] = np.transpose([XX, np.average(d1_densities, axis=0)])
        # results['density']['SURF-TRIO'] = np.transpose([XX, np.average(d2_densities, axis=0)])
        # results['density']['CORE-TRIO'] = np.transpose([XX, np.average(d3_densities, axis=0)])
        # results['density']['water'] = np.transpose([XX, np.average(d_water_densities, axis=0)])


        print("units: Z (nm), interdigitation (nm), time (ns), density (g/m3)")
        return results

    def save_results(self, results, base_dir):
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        for key1 in results.keys():
            new_dir = os.path.join(base_dir, key1)
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)

            for key2 in results[key1].keys():
                # Define new file path
                file_path = os.path.join(new_dir, f'interdigit_.{key1}.{key2}.dat')
                # Save the file
                np.savetxt(file_path, results[key1][key2])

        print(f"All files have been saved in the directory: {base_dir}")




class Membrane_overlap_distance:
    
    def __init__(self, universe, start_frame=0, end_frame=None):
        self.u = universe
        self.b = start_frame
        self.e = end_frame
        self.groups = {}
        
    
    def setup_groups(self, atom_name='C316', lipid_1='POPC', lipid_2='DOPE', lipid_3='SAPI', neutral_lipid='TRIO'):
        halfz = self.u.dimensions[2] / 2
        C2 = ' '.join([f'C2{i}' for i in range(2, 22)])
        C3 = ' '.join([f'C3{i}' for i in range(2, 22)])
        self.groups['upper_leaflet_atoms'] = self.u.select_atoms(f'resname {lipid_1} {lipid_2} and name {atom_name} and prop z > {halfz}')
        self.groups['lower_leaflet_atoms'] = self.u.select_atoms(f'resname {lipid_1} {lipid_2} and name {atom_name} and prop z < {halfz}')
        us = f'(same residue as resname {lipid_1} {lipid_2} {lipid_3} and name P and prop z>{halfz}) and name ' + C2 + ' ' + C3
        ls = f'(same residue as resname {lipid_1} {lipid_2} {lipid_3} and name P and prop z<{halfz}) and name ' + C2 + ' ' + C3
        self.groups['umemb'] = self.u.select_atoms(us)
        self.groups['lmemb'] = self.u.select_atoms(ls)
        self.groups['nl_positions'] = self.u.select_atoms(f'resname {neutral_lipid}')

    
    def calculate_average_z(self, atom_group):
        average_z = np.average(atom_group.positions[:, 2])
        return average_z
    
    def identify_strong_residues_all(self, nl_positions, utz, ltz):
        strong_resids = set()
        for residue in nl_positions.residues:
            oxygen_atoms = residue.atoms.select_atoms('name O11 O12 O21 O22 O31 O32 O10')
            o_positions = oxygen_atoms.positions[:, 2]
            if (o_positions > utz).all() or (o_positions < ltz).all():
                strong_resids.add(residue.resid)
        return strong_resids
    
    def identify_strong_residues_any(self, nl_positions, utz, ltz):
        strong_resids = set()
        for residue in nl_positions.residues:
            oxygen_atoms = residue.atoms.select_atoms('name O11 O12 O21 O22 O31 O32 O10')
            o_positions = oxygen_atoms.positions[:, 2]
            if (o_positions > utz).any() or (o_positions < ltz).any():
                strong_resids.add(residue.resid)
        return strong_resids
    
    def analyze_trajectory(self, method='all'):
        results = []
        for ts in self.u.trajectory[self.b:self.e]:
            timestep_results = []
            upper_C218_average = self.calculate_average_z(self.groups['upper_leaflet_atoms'])
            lower_C218_average = self.calculate_average_z(self.groups['lower_leaflet_atoms'])

            
            utz = np.average(self.groups['umemb'].positions[:, 2])
            ltz = np.average(self.groups['lmemb'].positions[:, 2])

            
            if method == 'all':
                strong_resids = self.identify_strong_residues_all(self.groups['nl_positions'], utz, ltz)
            elif method == 'any':
                strong_resids = self.identify_strong_residues_any(self.groups['nl_positions'], utz, ltz)

            for residue in self.groups['nl_positions'].residues:
                if residue.resid not in strong_resids:
                    z_positions = residue.atoms.positions[:, 2]
                    above_C218 = z_positions[z_positions > upper_C218_average]
                    below_C218 = z_positions[z_positions < lower_C218_average]

                    if above_C218.size > 0:
                        max_above = np.max(above_C218) - upper_C218_average
                        timestep_results.append((residue.resid, max_above))

                    if below_C218.size > 0:
                        max_below = lower_C218_average - np.min(below_C218)
                        timestep_results.append((residue.resid, max_below))

            results.append((ts.time, timestep_results))

        return results


    def analysis_to_df(self, analysis_results):
        data = {"Time": [], "Residue_ID": [], "Max_Overlap_Distance": []}
        for time, overlaps in analysis_results:
            for resid, distance in overlaps:
                data["Time"].append(time)
                data["Residue_ID"].append(resid)
                data["Max_Overlap_Distance"].append(distance)
        return pd.DataFrame(data)
    

class MicAnalysis:
    def __init__(self, lipid1='POPC', lipid2='DOPE', atom1='P', atom2='C316'):
        self.lipid1 = lipid1
        self.lipid2 = lipid2
        self.atom1 = atom1
        self.atom2 = atom2

    def thickness(self, u, startframe=0, endframe=None):
        halfz = u.dimensions[2] / 2
        average_per = []
        for ts in u.trajectory[startframe:endframe]:
            upper_P_atoms = u.select_atoms(f'resname {self.lipid1} {self.lipid2} and name {self.atom1} and prop z > {halfz}')
            lower_P_atoms = u.select_atoms(f'resname {self.lipid1} {self.lipid2} and name {self.atom1} and prop z < {halfz}')
            upper_C316_atoms = u.select_atoms(f'resname {self.lipid1} {self.lipid2} and name {self.atom2} and prop z > {halfz}')
            lower_C316_atoms = u.select_atoms(f'resname {self.lipid1} {self.lipid2} and name {self.atom2} and prop z < {halfz}')
            upper_distances = [np.abs(p.position[2] - c.position[2]) for p, c in zip(upper_P_atoms, upper_C316_atoms)]
            lower_distances = [np.abs(p.position[2] - c.position[2]) for p, c in zip(lower_P_atoms, lower_C316_atoms)]
            all_distances = upper_distances + lower_distances
            if all_distances:
                average_distance = np.mean(all_distances)
                average_per.append(average_distance)
            else:
                average_per.append(0)

        return average_per

    def MIC(self, depth, number, number_PL, membrane_thickness):
        mic = (depth*number)/number_PL
        rmic = (mic/membrane_thickness)*100
        return(rmic)

    def calculate_mic_analysis(self, u, overlap_overtime, overlap_number, total_phospholipids, startframe=0, endframe=None):
        tpf = self.thickness(u, startframe, endframe)
        df_combined = pd.merge(overlap_overtime, overlap_number, on='Time')
        df_combined['tpf'] = tpf
        df_combined['MIC'] = df_combined.apply(
            lambda row: self.MIC(row['Average_Overlap_Depth'], row['Number_of_Overlaps'], total_phospholipids, row['tpf']),
            axis=1
        )

        return df_combined

class OverlapAnalysis:
    
    def __init__(self, df, method='any'):
        self.df = df
        self.method = method
    
    def calculate_average_overlap_depth(self):
        # Group by 'Time' and calculate the mean of 'Max_Overlap_Distance'
        average_overlaps = self.df.groupby('Time')['Max_Overlap_Distance'].mean().reset_index()
        average_overlaps.columns = ['Time', 'Average_Overlap_Depth']
        return average_overlaps
    
    def calculate_number_of_overlaps_per_frame(self):
        # Count occurrences of each 'Time' value
        overlaps_count = self.df.groupby('Time').size().reset_index(name='Number_of_Overlaps')
        return overlaps_count
    
    def calculate_average_depth_overall(self):
        total_average = self.df['Max_Overlap_Distance'].mean()
        return total_average
    
    def print_analysis(self):
        overlaps = self.calculate_number_of_overlaps_per_frame()
        average_overlaps = self.calculate_average_overlap_depth()
        overall_average = self.calculate_average_depth_overall()
        
        #print(f"Results for {self.method}:")
        #print("Number of Overlaps per Frame:", overlaps)
        #print("Average Overlaps Depth per Frame:", average_overlaps)
        print("Overall Average Overlap Depth:", overall_average)

