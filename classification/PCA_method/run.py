import pandas as pd 
import numpy as np #use numpy 1.20
import subprocess
import os
import conntility
import connalysis


database_root = './classification/PCA/'

if not os.path.isfile(database_root+'community_database.pkl'):
    print("Downloading and extracting structural data")
    zenodo_address = 'https://zenodo.org/records/10812497'
    os.system("wget "+zenodo_address+"/files/classification.xz")
    os.system("tar -xf classification.xz") 
    #os.system("unzip community_database.zip")
    print("structural data ready")

### Uncomment this section to create community database from scratch using selections.pkl
# #Load dataframe which says what the top parameters are for double selection
# A=pd.read_pickle('selections.pkl')
#Load create and then load dataframe
# _=subprocess.call(['python', './topological_sampling/pipeline/gen_topo_db/gen_topo_db.py', './topological_sampling/working_dir/config/common_config.json', 'tribe'])
# _=subprocess.call(['python', './topological_sampling/pipeline/gen_topo_db/gen_topo_db.py', './topological_sampling/working_dir/config/common_config.json', 'neuron_info'])
# _=subprocess.call(['python', './topological_sampling/pipeline/gen_topo_db/merge_database.py', './topological_sampling/working_dir/config/common_config.json'])
# D=pd.read_pickle('./topological_sampling/working_dir/data/analyzed_data/community_database.pkl').iloc[:,:7]
# n = len(D)

# #Create the names of all the second selections
# names = []
# indices = []
# for i in range(len(A)):
#     if pd.isnull(A['second_selection'][i]):
#     #if pd.isnull(A['first_selection'][i]):
#         #pass
#         #names.append(A['second_selection'][i]+'_'+A['selection_order'][i])
#         names.append(A['first_selection'][i]+'_'+A['selection_order'][i])
#         indices.append(i)
#     else:
#         names.append(A['first_selection'][i]+'_then_'+A['second_selection'][i]+'_'+A['selection_order'][i])
#         indices.append(i)

# #Create a map that takes an index in the exc-exc circuit to its index in the full circuit
# D_reset = D.reset_index()
# exc_to_all_index = D_reset[D_reset['synapse_class']=='EXC'].index

# #Add a column to the dataframe which is 1 for the top selected tribes, and 0 for all other tribes
# #Do this for each element of names
# for i in range(len(names)):
#     Z = np.zeros(n,dtype='uint8')
#     Z[[exc_to_all_index[j] for j in list(A.iloc[indices[i]])[3:]]]=1
#     D[names[i]] = Z

# #Save the new dataframe
# D.to_pickle(database_root+'community_database.pkl')

##############################################################################################

os.system("mkdir topological_sampling/working_dir/data")
os.system("mkdir topological_sampling/working_dir/data/analyzed_data")

#Load dataframe which says what the top parameters are for double selection
D=pd.read_pickle(database_root+'community_database_PCA.pkl')
n = len(D)

#Create the names of all the second selections
names = D.keys()
indices = list(range(len(names)))

if not os.path.isfile("./topological_sampling/working_dir/data/input_data/raw_spikes.npy"):
    print("Downloading and extracting simulation data")
    os.system("wget "+zenodo_address+"/files/simulation.xz")
    os.system("tar -xvf simulation.xz BlobStimReliability_O1v5-SONATA_Baseline/working_dir") 
    os.system("mv ./BlobStimReliability_O1v5-SONATA_Baseline/working_dir ./topological_sampling/working_dir/data/input_data") 
    os.system("mv ./topological_sampling/working_dir/data/input_data/raw_spikes_exc_0.npy ./topological_sampling/working_dir/data/input_data/raw_spikes.npy") 
    print("Simulation data ready")

# Get connectivity object neighborhoods
connectome=conntility.ConnectivityMatrix.from_h5('./topological_sampling/working_dir/data/input_data/connectome.h5')
nbds=connalysis.network.local.neighborhood_indices(connectome.matrix, all_nodes=True)
D['tribe']=list(connalysis.network.local.neighborhood_indices(connectome.matrix, all_nodes=True))
D.to_pickle('./topological_sampling/working_dir/data/analyzed_data/community_database.pkl')
ss.save_npz('./topological_sampling/working_dir/data/input_data/connectivity.npz',connectome.matrix.tocsr())




#Create the   "Champions">"Specifiers" entry for samping_config.json
if not os.path.isfile('./topological_sampling/working_dir/config/sampling_config_orig.json'):
    os.rename('./topological_sampling/working_dir/config/sampling_config.json','./topological_sampling/working_dir/config/sampling_config_orig.json')

f = list(open('./topological_sampling/working_dir/config/sampling_config_orig.json'))
g = open('./topological_sampling/working_dir/config/sampling_config.json','w')
for i in range(74):
    g.write(f[i])

for i in range(len(names)):
      g.write('      {\n')
      g.write('        "name": "'+names[i]+'",\n')
      g.write('        "value": {\n')
      g.write('          "column": "'+names[i]+'"\n')
      g.write('        },\n')
      g.write('        "number": 50\n')
      if i == len(names)-1:
          g.write('      }\n')
      else:
          g.write('      },\n')


for i in range(286,len(f)):
    g.write(f[i])

g.close()


def opening_sbatch(g):
      g.write('#!/bin/bash\n')
      #g.write('#SBATCH -N 1\n')
      #g.write('#SBATCH -n 36\n')
      #g.write('#SBATCH --mem 300G\n')
      #g.write('#SBATCH -o ./working_dir/out/%j.out\n')
      #g.write('#SBATCH -e ./working_dir/err/%j.err\n')
      #g.write('#SBATCH --time=24:00:00\n')
      #g.write('#SBATCH --job-name=topsamp\n')
      #g.write('#SBATCH --account=proj9\n')
      #g.write('#SBATCH --partition=prod\n')
      #g.write('source /gpfs/bbp.cscs.ch/project/proj102/smith/topological_sampling_rel/topsamp/bin/activate\n')


g = open('run_manifold_analysis_param.sh','w')
opening_sbatch(g)
for j in range(len(names)):
     g.write('python ./topological_sampling/pipeline/manifold_analysis/manifold_analysis.py ./topological_sampling/working_dir/config/common_config.json components "sampling=Parameter" "specifier='+names[j]+'"\n')

g.close()


reps = 6
for typ in ['classifier']:
    for i in range(reps):
        g = open('run_'+typ+'_param'+str(i)+'.sh','w')
        opening_sbatch(g)
        end = (i+1)*int(len(names)/reps)
        if i == reps-1:
            end = len(names)
        for j in range(i*int(len(names)/reps),end):
            g.write('python ./topological_sampling/pipeline/'+typ+'/'+typ+'.py ./topological_sampling/working_dir/config/common_config.json components "sampling=Parameter" "specifier='+names[j]+'"\n')

        g.close()


print("\n\n\n\n\nPipeline is ready to be run, please execute:")
print("python ./topological_sampling/pipeline/sample_tribes/sample-tribes-champions.py ./topological_sampling/working_dir/config/common_config.json")

print("\nThen run all the following:")
print("sh run_manifold_analysis_param.sh")


print("\nThen run all of ")
for i in range(reps):
    print("sh run_classifier_param"+str(i)+".sh")

print("\nFinally run:")
print("python create_results_df.py")


print("\nThe results will be in the pandas dataframe classification_results.pkl")


####DATABASE needs the tribes.
