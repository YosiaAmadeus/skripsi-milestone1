import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting

import logging
from flask import Flask, request, render_template, redirect, url_for, flash, send_file
import pandas as pd
from werkzeug.utils import secure_filename
import os
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import skfuzzy as fuzz
import numpy as np
import matplotlib.patches as mpatches

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'supersecretkey'

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Fungsi untuk menjalankan K-Means
def run_kmeans(data, clusters):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.values)
    kmeans = KMeans(n_clusters=clusters, random_state=42)
    data['Cluster'] = kmeans.fit_predict(scaled_data)
    return data

# Fungsi untuk menjalankan Fuzzy C-Means
def run_fuzzy_cmeans(data, clusters):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.values)
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(scaled_data.T, clusters, 1.5, error=0.001, maxiter=5000, init=None)
    cluster_membership = np.argmax(u, axis=0)
    data['Cluster'] = cluster_membership
    return data

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/riceplant101')
def riceplant101():
    return render_template('riceplant101.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/run-clustering', methods=['GET', 'POST'])
def run_clustering():
    if request.method == 'POST':
        if 'dataset' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['dataset']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                logging.debug('Membaca file dataset')
                data = pd.read_excel(filepath, index_col='Lokasi')
                clusters = int(request.form['clusters'])
                algorithm = request.form['algorithm']
                logging.debug(f'Jumlah cluster yang dipilih: {clusters}')
                logging.debug(f'Algoritma yang dipilih: {algorithm}')

                # Deteksi skala data
                if any(data.index.str.contains('Kabupaten|Kota', case=False)):
                    logging.debug('Menggunakan shapefile level 2 (Kabupaten/Kota)')
                    shapefile_path = "gadm41_IDN_2.shp"
                    merge_on = "NAME_2"
                else:
                    logging.debug('Menggunakan shapefile level 1 (Provinsi)')
                    shapefile_path = "gadm41_IDN_1.shp"
                    merge_on = "NAME_1"

                # Pastikan algoritma yang dipilih dijalankan dengan benar
                if algorithm == 'kmeans':
                    logging.debug('Menjalankan model K-Means')
                    result = run_kmeans(data, clusters)
                elif algorithm == 'fuzzy':
                    logging.debug('Menjalankan model Fuzzy C-Means')
                    result = run_fuzzy_cmeans(data, clusters)

                # Buat visualisasi distribusi cluster
                logging.debug('Membuat visualisasi distribusi cluster')
                cluster_means = result.groupby('Cluster').mean()
                sorted_clusters = cluster_means.mean(axis=1).sort_values().index.tolist()

                color_map = {}
                legend_patches = []
                
                if clusters == 2:
                    color_map = {sorted_clusters[0]: 'red', sorted_clusters[1]: 'blue'}
                    legend_patches = [
                        mpatches.Patch(color='red', label=f'Cluster {sorted_clusters[0]} (Rendah)'),
                        mpatches.Patch(color='blue', label=f'Cluster {sorted_clusters[1]} (Tinggi)')
                    ]
                    colors = ['red', 'blue']
                else:
                    color_map = {sorted_clusters[0]: 'red', sorted_clusters[1]: 'orange', sorted_clusters[2]: 'blue'}
                    legend_patches = [
                        mpatches.Patch(color='red', label=f'Cluster {sorted_clusters[0]} (Rendah)'),
                        mpatches.Patch(color='orange', label=f'Cluster {sorted_clusters[1]} (Sedang)'),
                        mpatches.Patch(color='blue', label=f'Cluster {sorted_clusters[2]} (Tinggi)')
                    ]
                    colors = ['red', 'orange', 'blue']

                cluster_distribution = result['Cluster'].value_counts().reindex(sorted_clusters)
                cluster_distribution.plot(kind='bar', color=[color_map[c] for c in sorted_clusters])
                plt.title('Plot Distribusi Cluster')
                plt.xlabel('Cluster')
                plt.ylabel('Jumlah Provinsi')
                plt.grid(True)
                plt.savefig('static/distribution.png')
                plt.close()

                # Kamus penyesuaian nama provinsi
                kamus_penyesuaian = {
                    "Daerah Khusus Ibukota Jakarta": "Jakarta Raya",
                    "Kepulauan Bangka Belitung": "Bangka Belitung",
                    "Daerah Istimewa Yogyakarta": "Yogyakarta",
                }
                
                # Muat shapefile Indonesia
                logging.debug(f'Muat shapefile {shapefile_path}')
                gdf = gpd.read_file(shapefile_path)
                data_clustering = result[['Cluster']].reset_index().rename(columns={'Lokasi': 'provinsi'})
                
                # Terapkan kamus penyesuaian
                data_clustering['provinsi'] = data_clustering['provinsi'].replace(kamus_penyesuaian)
                
                gdf_provinsi = gdf.dissolve(by=merge_on)
                gdf_provinsi = gdf_provinsi.merge(data_clustering, left_on=merge_on, right_on='provinsi', how='left')
                gdf_provinsi['color'] = gdf_provinsi['Cluster'].map(color_map).fillna('lightgrey')

                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                gdf_provinsi.plot(ax=ax, color=gdf_provinsi['color'], alpha=0.7, edgecolor='black')
                ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                for x, y, label in zip(gdf_provinsi.geometry.centroid.x, gdf_provinsi.geometry.centroid.y, gdf_provinsi['provinsi']):
                    ax.text(x, y, label, fontsize=8, ha='center', va='center', color='black')
                plt.legend(handles=legend_patches, loc='upper right')
                plt.title("Visualisasi Hasil Clustering Produktivitas Padi di Indonesia")
                plt.xlabel("Longitude")
                plt.ylabel("Latitude")
                plt.savefig('static/map.png')
                plt.close()

                # Simpan hasil clustering ke file Excel
                output = result[['Cluster']].reset_index().rename(columns={'Lokasi': 'Daerah'})
                output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.xlsx')
                output.to_excel(output_path, index=False)
                flash('Download output.xlsx')

            except Exception as e:
                logging.error('Error: %s', e)
                flash('An error occurred while processing the data.')
            finally:
                logging.debug('Menghapus file sementara')
                # Hapus file setelah digunakan
                if os.path.exists(filepath):
                    os.remove(filepath)

            return render_template('index.html', result=True)
    return redirect(url_for('home'))

@app.route('/download-excel')
def download_excel():
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.xlsx')
    return send_file(output_path, as_attachment=True)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)