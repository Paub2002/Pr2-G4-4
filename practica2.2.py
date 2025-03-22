import cv2
import metrikz
import utility
import pylab
import matplotlib.pyplot as plt

def count_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames

if __name__ == '__main__':
	# Noms dels arxius d'entrada i sortida.
	video_files = ['akiyo_cif', 'bus_cif', 'flower_cif','football_cif','foreman_cif']

	# Establim els parametres de q-scale que farem servir
	q_values = ['5','10','15','25']

	process_videos = False

	if process_videos:
		# Recorrem tots els videos i els comprimim utilitzant els 4 parametres de q-scale
		for video_file in video_files:

			for q in q_values:
				video_input_file = 'Videos/' + video_file + '.y4m'
				video_output_file = 'Videos/' + video_file + f'_{q}_' + 'H261' + '.avi'
				
				# Comanda per a la compresio a H.261
				command = [
					'ffmpeg',
					'-y',
					'-an',
					'-i', video_input_file, 
					'-q:v', q,
					'-vcodec', 'h261',
					video_output_file,
				]

				utility.execute_command(command)
				
				video_output_file = 'Videos/' + video_file + f'_{q}_' + 'MPEG2' + '.avi'
				
				# Comanda per a la compresio a MPEG2
				command = [
					'ffmpeg',
					'-y',
					'-an',
					'-i', video_input_file, 
					'-q:v', q,
					'-vcodec', 'mpeg2video',
					video_output_file,
				]

				# Executem la comanda
				utility.execute_command(command)

			print(f"File {video_file}")

		# A continuació generem els frames per cada video original
		for video_file in video_files:
			# Nom del video
			video_input_file = 'Videos/' + video_file + '.y4m'

			# Comanda per a extraure els quadres (frames) del video original
			command = [
				'ffmpeg',
				'-y',
				'-i', video_input_file,
				'./frames/' + f'{video_file}/' + 'original' + f'_{video_file}_' + '%04d.png', # Amb %04 assegurem que tots tenen el mateix format
			]

			utility.execute_command(command)        

			for q in q_values:
				# Nom del arxiu de sortida en funció de la q i compressió H261
				video_output_file = 'Videos/' + video_file + f'_{q}_' + 'H261' + '.avi'

				# Comanda per a extreure els quadres del video codificat
				command = [
					'ffmpeg',
					'-y',
					'-i', video_output_file,
					'./frames/' + f'{video_file}/' + 'encoded' + f'_{video_file}_{q}_H261_' + '%04d.png', 
				]

				utility.execute_command(command)
			
				# Nom del arxiu de sortida en funció de la q i compressió MPEG2
				video_output_file = 'Videos/' + video_file + f'_{q}_' + 'MPEG2' + '.avi'

				# Comanda per a extreure els quadres del video codificat
				command = [
					'ffmpeg',
					'-y',
					'-i', video_output_file,
					'./frames/' + f'{video_file}/' + 'encoded' + f'_{video_file}_{q}_MPEG2_' + '%04d.png', 
				]

				utility.execute_command(command)

	
	print("MSE mean: ")
	for video_file in video_files:
		print(f"\n{video_file.upper()}")
		n_frames = count_frames('Videos/' + video_file + '.y4m')

		for q in q_values:
			mse_mean_H261 = 0
			mse_mean_MPEG2 = 0
			
			for i in range(n_frames):
				source = cv2.imread('./frames/' + f'{video_file}/' + 'original' + f'_{video_file}_' + str(i+1).zfill(4) + '.png')
				target_H261 = cv2.imread('./frames/' + f'{video_file}/' + 'encoded' + f'_{video_file}_{q}_H261_' + str(i+1).zfill(4) + '.png')
				target_MPEG2 = cv2.imread('./frames/' + f'{video_file}/' + 'encoded' + f'_{video_file}_{q}_MPEG2_' + str(i+1).zfill(4) + '.png')

				mse_mean_H261 += metrikz.mse(source, target_H261)
				mse_mean_MPEG2 += metrikz.mse(source, target_MPEG2)

			mse_mean_H261 /= n_frames
			mse_mean_MPEG2 /= n_frames

			print(f"Q-scale ===> {q}")
			print(f"MSE mean for H261 : {mse_mean_H261}")
			print(f"MSE mean for MPEG2 : {mse_mean_MPEG2}")
