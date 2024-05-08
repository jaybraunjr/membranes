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

    def interdigit(self, nbins=50, nblocks=5, b=0, e=None):
        
        halfz = self.u.dimensions[2] / 2
        numP = self.u.select_atoms('name P').n_atoms

        # Generating the selection strings for C2 and C3 atoms
        C2 = ' '.join([f'C2{i}' for i in range(2, 22)])
        C3 = ' '.join([f'C3{i}' for i in range(2, 22)])
        tail_atoms = ' '.join(C2.split() + C3.split())  # Merging C2 and C3 atom names

        lipid_selection = ' or '.join([f'resname {lipid}' for lipid in self.lipids])
        us = f'(same residue as ({lipid_selection}) and name P and prop z>{halfz}) and (name {" ".join(tail_atoms.split())})'
        ls = f'(same residue as ({lipid_selection}) and name P and prop z<{halfz}) and (name {" ".join(tail_atoms.split())})'

        
        # C2 = ' '.join(['C2%d' % i for i in range(2, 22)])
        # C3 = ' '.join(['C3%d' % i for i in range(2, 22)])
        

        # us = '(same residue as resname {} and name P and prop z>{}) and name '.format(self.lipids, halfz) + C2 + C3
        # ls = '(same residue as resname {} and name P and prop z<{}) and name '.format(self.lipids, halfz) + C2 + C3
        # lipid_selection = ' or '.join(['resname {}'.format(lipid) for lipid in self.lipids])
    
        groups = {'memb':  self.u.select_atoms(lipid_selection),
                  'umemb': self.u.select_atoms(us),
                  'lmemb': self.u.select_atoms(ls),
                  'trio': self.u.select_atoms(f'resname {self.NL}')}
        groups['water'] =  self.u.select_atoms(f'resname {self.water}')
        print(groups)

        assert groups['umemb'].n_atoms == groups['lmemb'].n_atoms, "# of umemb atoms != # of lmemb atoms"

        names = groups['trio'].names.astype(str)
        resids = groups['trio'].resids
        
        times = []
        zs = []
        total_inter = []
        strong_inter = []
        weak_inter = []
        
        d0_densities = []  # PL
        d1_densities = []  # TRIO
        d2_densities = []  # SURF-TRIO
        d3_densities = []  # CORE-TRIO

        total_ov = []
        strong_ov = []
        weak_ov = []
        
        strong_num = []

        for ts in self.u.trajectory[b:e]:
            if int(ts.time / 1000) % 1000 == 0:
                print("analyzing %d us.... " % (ts.time / 1000 / 1000))

            pbc = self.u.dimensions
            bins = np.linspace(0, pbc[2], nbins + 1)
            dz = bins[1] - bins[0]
            trio_pos = groups['trio'].positions

            utz = np.average(groups['umemb'].positions[:, 2])
            ltz = np.average(groups['lmemb'].positions[:, 2])
            
            d0 = self.density_frame(groups['memb'].positions[:, 2], groups['memb'].masses, pbc=pbc, bins=bins)
            d0_densities.append(d0)
            
            d1 = self.density_frame(groups['trio'].positions[:, 2], groups['trio'].masses, pbc=pbc, bins=bins)
            d1_densities.append(d1)

            d_water_densities = []
            d_water = self.density_frame(groups['water'].positions[:, 2], groups['water'].masses, pbc=pbc, bins=bins)
            d_water_densities.append(d_water)

            tov = self.cal_overlap(d0, d1)
            tin = self.cal_inter(tov, dz)
            total_ov.append(tov)
            total_inter.append(tin)

            boolArray = ((trio_pos[:, 2] > utz) | (trio_pos[:, 2] < ltz)) & np.char.startswith(names, 'O')
            strong_resids = resids[boolArray]
            r = np.unique(strong_resids, return_counts=True)
            boolArray = (r[1] == 6)
            strong_resids = r[0][boolArray]
            boolArray = np.isin(resids, strong_resids)
            pp = trio_pos[boolArray]
            mm = groups['trio'].masses[boolArray]

            d2 = self.density_frame(pp[:, 2], mm, pbc, bins)
            d2_densities.append(d2)

            sov = self.cal_overlap(d0, d2)
            sin = self.cal_inter(sov, dz)
            strong_ov.append(sov)
            strong_inter.append(sin)

            boolArray = np.isin(resids, strong_resids, invert=True)
            pp = trio_pos[boolArray]
            mm = groups['trio'].masses[boolArray]

            d3 = self.density_frame(pp[:, 2], mm, pbc, bins)
            d3_densities.append(d3)

            wov = self.cal_overlap(d0, d3)
            win = self.cal_inter(wov, dz)
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

        results['density'] = {}
        results['density']['PL'] = np.transpose([XX, np.average(d0_densities, axis=0)])
        results['density']['TRIO'] = np.transpose([XX, np.average(d1_densities, axis=0)])
        results['density']['SURF-TRIO'] = np.transpose([XX, np.average(d2_densities, axis=0)])
        results['density']['CORE-TRIO'] = np.transpose([XX, np.average(d3_densities, axis=0)])
        results['density']['water'] = np.transpose([XX, np.average(d_water_densities, axis=0)])

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
