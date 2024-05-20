#Script para automatizar el programa de indice de metadiversidad de python
#Es recomendable respaldar el MD_results.csv que ya este creado porque agrega entradas entonces puede haber redundancia

#rm -f MD_results.csv
touch MD_results.csv
echo "SAMPLE,CONTIGS,RICHNESS,SHANNON,SIMPSON,LOG10_MD,MDI" > ./MD_results.csv

for filename in /files/camda2024/gut/prokka/*.faa; do
	python metadiversity_index.py -i $filename -r F
	done

echo "Trabajo Finalizado"
