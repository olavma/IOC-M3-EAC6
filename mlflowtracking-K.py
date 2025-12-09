""" @ IOC - Olav Martos - 2025 - CE IABD """

import logging
import shutil
import mlflow
from mlflow.tracking import MlflowClient
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("..")
from clustersciclistes import load_dataset, clean, extract_true_labels, clustering_kmeans, homogeneity_score, completeness_score, v_measure_score

# Carpeta mlruns/
project_dir = os.path.dirname(os.path.abspath(__file__))
mlruns_dir = os.path.join(project_dir, "mlruns")
mlflow.set_tracking_uri(f"file://{mlruns_dir}")

# Client MLflow, si no es queixa
client = MlflowClient()

def get_run_dir(artifacts_uri):
	""" retorna ruta del run """
	return artifacts_uri.replace("file://", "").replace("/artifacts", "")
	
def remove_run_dir(run_dir):
	""" elimina path amb shutil.rmtree """
	shutil.rmtree(run_dir, ignore_errors=True)

if __name__ == "__main__":
	logging.basicConfig(format='%(message)s', level=logging.INFO) # canviar entre DEBUG i INFO
	
	experiment_name = "K sklearn ciclistes"
	mlflow.set_experiment(experiment_name)
	exp = client.get_experiment_by_name(experiment_name)
	
	client.set_experiment_tag(exp.experiment_id, "version", "1.0")
	client.set_experiment_tag(exp.experiment_id, "scikit-learn", "K")
	client.set_experiment_tag(exp.experiment_id, "mlflow.note.content", "ciclistes variació de paràmetre K")
	
	runs = client.search_runs([exp.experiment_id])

	# Esborrem tots els runs de l'experiment
	for run in runs:
		mlflow.delete_run(run.info.run_id)
		remove_run_dir(get_run_dir(run.info.artifact_uri))
	
	path_dataset = './data/ciclistes.csv'
	ciclistes_data = load_dataset(path_dataset)
	ciclistes_data = clean(ciclistes_data)
	true_labels = extract_true_labels(ciclistes_data)
	ciclistes_data = ciclistes_data.drop('tipus', axis=1)
	
	Ks = [2, 3, 4, 5, 6, 7, 8]

	for K in Ks:
		dataset_path = f'./data/ciciclistes_K{K}.csv'
		ciclistes_data.to_csv(dataset_path, index=False)
		
		with mlflow.start_run(description=f"K={K}"):
			clustering_model = clustering_kmeans(ciclistes_data, K)
			data_labels = clustering_model.labels_
			h_score = round(homogeneity_score(true_labels, data_labels), 5) 
			c_score = round(completeness_score(true_labels, data_labels), 5)
			v_score = round(v_measure_score(true_labels, data_labels), 5)
			
			logging.info('K: %d', K)
			logging.info('H-measure: %.5f', h_score)
			logging.info('C-measure: %.5f', c_score)
			logging.info('V-measure: %.5f', v_score)

			mlflow.set_tags({
				"engineering": "OMA-IOC",
                "release.candidate": "RC1",
                "release.version": "1.1.2",
			})
			mlflow.log_param("K", K)
			mlflow.log_metric("h", h_score)
			mlflow.log_metric("c", c_score)
			mlflow.log_metric("v", v_score)
			
			mlflow.log_artifact(dataset_path)
			
			os.makedirs("img", exist_ok=True)
			fig = plt.figure()
			sns.scatterplot(x='temps_pujada', y='temps_baixada', data=ciclistes_data, hue=data_labels, palette="rainbow")
			plt.savefig(f"img/olav_martos_grafica_K{K}.png")
			fig.clf()

	print('s\'han generat els runs')
