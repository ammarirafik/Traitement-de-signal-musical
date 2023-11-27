import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np #faire une fenêtre de hamming et multiplication par le signal d'origine
import soundfile as sf
from scipy.fft import fft
from scipy.signal import spectrogram
from numpy.random import normal
from scipy.io.wavfile import write
from scipy.signal import wiener
import os
from pydub import AudioSegment // #manque de ffmpeg que nous avons telechargé, ainsi pour Pydub utilise FFmpeg pour lire et écrire des fichiers audio dans divers formats tels que MP3, WAV, FLAC, etc.
from pydub.effects import compress_dynamic_range

fichier_audio = "C:/Users/ASUS/PycharmProjects/pythonProject1/base de donnée/485084__matt141141__manouche_guitar_loop2.wav"

# Créez une fenêtre principale de l'interface graphique
root = tk.Tk() #Crée une fenêtre principale pour l'interface graphique.
root.title("Signal musical")
signal_fenetree = None

# Utilisation de la même figure principale et canvas dans toutes les fonctions (collaboration entre matplotlib et tkinter)
fig, ax = plt.subplots(figsize=(2, 1))
fig.patch.set_facecolor('#F0F0F0')
fig.canvas.draw() #canvas c'est une zone graphique
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
fig_spectrogram = None

# Charger le fichier à afficher
signal_audio_a_afficher, sample_rate = sf.read(fichier_audio)
temps = np.arange(len(signal_audio_a_afficher)) / sample_rate

# Initialisation de l'index pour suivre la position actuelle
index = 0

# Variable globale pour vérifier si l'acquisition est en cours
acquisition_en_cours = False

# Variable pour indiquer si le signal doit être affiché
afficher_signal = False

# Définir segment_length (ajustez cette valeur selon vos besoins)
segment_length = 2000  # ou toute autre valeur appropriée

# Fonction pour fenêtrer le signal et l'enregistrer dans la variable globale
def fenetrer_signal():
    global signal_fenetree, afficher_signal, fig_fenetree, ax_fenetree, canvas_fenetree
    # Effacer le contenu de la figure du spectre existante
    if fig_spectre is not None:
        fig_spectre.clf()
        canvas_spectre.get_tk_widget().destroy()
        # Créer une nouvelle figure pour le signal fenêtré bruité
    if fig_fenetree is not None:
        fig_fenetree.clf()
        canvas_fenetree.get_tk_widget().destroy()

    # Trouver les indices correspondant à 0S et 5S
    indice_debut = int(0 * sample_rate)
    indice_fin = int(5 * sample_rate)

    # Extraire le segment du signal entre 0S et 5S
    signal_fenetree = apply_window(signal_audio_a_afficher[indice_debut:indice_fin])


    fig_fenetree, ax_fenetree = plt.subplots(figsize=(2, 1))
    fig_fenetree.patch.set_facecolor('#F0F0F0')  # Couleur de fond du graphique
    fig_fenetree.canvas.draw()
    canvas_fenetree = FigureCanvasTkAgg(fig_fenetree, master=root)
    canvas_widget_fenetree = canvas_fenetree.get_tk_widget()
    canvas_widget_fenetree.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Afficher le signal fenêtré dans la nouvelle figure
    ax_fenetree.plot(temps[indice_debut:indice_fin], signal_fenetree)
    ax_fenetree.set_xlabel('Temps (s)')
    ax_fenetree.set_ylabel('Amplitude')
    ax_fenetree.set_title('Signal musical (Fenêtré de 0S à 5S)', fontsize=10, fontweight='bold', fontstyle='italic', color='blue')
    ax_fenetree.legend(["Signal Fenêtré"])
    canvas_fenetree.draw()

# Fonction pour ajouter du bruit au signal (définition du bruit )
def ajouter_bruit(signal, amplitude_bruit):
    bruit = normal(0, amplitude_bruit, len(signal))
    signal_bruite = signal + bruit
    return signal_bruite

# Fonction pour ajouter du bruit au signal et mettre à jour le graphique (ajout au siganl actuelle)
def ajout_bruit():
    global signal_audio_a_afficher, signal_fenetree, afficher_signal, fig_fenetree, ax_fenetree, canvas_fenetree

    amplitude_bruit = 0.1  # Ajustez cette valeur selon vos besoins

    # Ajouter du bruit au signal
    signal_audio_a_afficher = ajouter_bruit(signal_audio_a_afficher, amplitude_bruit)

    # Mettre à jour le signal fenêtré si nécessaire
    if signal_fenetree is not None:
        signal_fenetree = ajouter_bruit(signal_fenetree, amplitude_bruit)

        # Effacer le contenu de la figure du spectre existante
        if fig_spectre is not None:
            fig_spectre.clf()
            canvas_spectre.get_tk_widget().destroy()
        if fig_spectrogram is not None:
            fig_spectrogram.clf()
            canvas_spectrogram.get_tk_widget().destroy()
        # Créer une nouvelle figure pour le signal fenêtré bruité
        if fig_fenetree is not None:
            fig_fenetree.clf()
            canvas_fenetree.get_tk_widget().destroy()

        fig_fenetree, ax_fenetree = plt.subplots(figsize=(5, 1))
        fig_fenetree.patch.set_facecolor('#F0F0F0')  # Couleur de fond du graphique
        fig_fenetree.canvas.draw()
        canvas_fenetree = FigureCanvasTkAgg(fig_fenetree, master=root)
        canvas_widget_fenetree = canvas_fenetree.get_tk_widget()
        canvas_widget_fenetree.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Afficher le signal fenêtré bruité dans la nouvelle figure
        ax_fenetree.plot(temps[:len(signal_fenetree)], signal_fenetree)
        ax_fenetree.set_xlabel('Temps (s)')
        ax_fenetree.set_ylabel('Amplitude')
        ax_fenetree.set_title('Signal musical fenêtré avec bruit', fontsize=10, fontweight='bold', fontstyle='italic', color='blue')
        ax_fenetree.legend(["Signal fenêtré avec bruit"])
        canvas_fenetree.draw()
def equalize_signal(signal):
    # Normalisation des amplitudes par fréquence
    return signal / np.max(np.abs(signal))
def update_equalized_graph(equalized_signal):
    global ax, canvas, temps

    # Limiter la longueur de temps à la longueur de equalized_signal
    temps_limite = temps[:len(equalized_signal)]

    # Mettre à jour le graphique dans le domaine temporel avec le signal égalisé
    ax.clear()
    ax.plot(temps_limite, equalized_signal)
    ax.set_xlabel('Temps (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Signal musical égalisé', fontsize=10, fontweight='bold', fontstyle='italic', color='blue')
    ax.legend(["Signal égalisé"])
    canvas.draw()
def initialize_spectrogram_figure():
    global fig_spectrogram, ax_spectrogram, canvas_spectrogram
    # Effacer le contenu de la figure existante
    if fig_spectre is not None:
        fig_spectre.clf()
        canvas_spectre.get_tk_widget().destroy()
        # Créer une nouvelle figure pour le signal fenêtré filtré
    if fig_fenetree is not None:
        fig_fenetree.clf()
        canvas_fenetree.get_tk_widget().destroy()
# Créer une nouvelle figure pour le spectrogramme
    fig_spectrogram, ax_spectrogram = plt.subplots(figsize=(3, 1))
    fig_spectrogram.patch.set_facecolor('#F0F0F0')  # Couleur de fond du graphique
    fig_spectrogram.canvas.draw()
    canvas_spectrogram = FigureCanvasTkAgg(fig_spectrogram, master=root)
    canvas_widget_spectrogram = canvas_spectrogram.get_tk_widget()
    canvas_widget_spectrogram.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

def apply_equalization():
    global signal_fenetree

    # Limiter le signal aux échantillons entre 0 et 5 secondes
    signal_partiel = signal_fenetree[:int(5 * sample_rate)]

    # Afficher le signal entre 0 et 5 secondes (ajout de cette ligne)
    plt.figure(figsize=(6, 2))
    plt.plot(signal_partiel)
    plt.title('Signal entre 0 et 5 secondes')
    plt.xlabel('Échantillons')
    plt.ylabel('Amplitude')
    plt.show()

    # Égaliser l'ensemble du signal entre 0 et 5 secondes
    equalized_signal = equalize_signal(signal_partiel)

    # Mettre à jour le graphique dans le domaine temporel
    update_equalized_graph(equalized_signal)

    # Initialiser la figure du spectrogramme après égalisation
    initialize_spectrogram_figure()

    # Afficher le spectrogramme du signal égalisé
    plot_spectrogram(equalized_signal)


def plot_spectrogram(signal):
    global ax_spectrogram, canvas_spectrogram
    # Calculer le spectrogramme
    f, t, Sxx = spectrogram(signal)
    # Mettre à jour le graphique du spectrogramme
    ax_spectrogram.clear()
    ax_spectrogram.pcolormesh(t, f, 10 * np.log10(Sxx), shading='auto', cmap='plasma')  # Remplacez 'viridis' par 'plasma'
    ax_spectrogram.set_xlabel('Temps (s)')
    ax_spectrogram.set_ylabel('Fréquence (Hz)')
    ax_spectrogram.set_title('Spectrogramme du signal égalisé', fontsize=9, fontweight='bold', fontstyle='italic', color='blue')
    canvas_spectrogram.draw()
def compress_signal_mp3(signal, threshold=-20.0, ratio=4.0, sample_width=2):
    # Convertir le signal en tableau NumPy avec une taille d'échantillon de 16 bits (2 bytes)
    signal_16bit = (signal * (2 ** 15)).astype(np.int16)

    # Convertir le signal en objet AudioSegment
    audio_segment = AudioSegment(signal_16bit.tobytes(), frame_rate=sample_rate, sample_width=sample_width, channels=1)

    # Appliquer la compression avec pydub
    compressed_audio = compress_dynamic_range(audio_segment, threshold=threshold, ratio=ratio)

    # Convertir le signal compressé en tableau NumPy
    compressed_signal = np.array(compressed_audio.get_array_of_samples())

    # Afficher le signal avant compression
    print("Signal avant compression:", signal)

    # Afficher le signal compressé
    print("Signal compressé:", compressed_signal)

    # Mettre à jour le graphique dans le domaine temporel
    update_after_compression(compressed_signal)

    # Sauvegarder le signal compressé en format WAV temporaire
    write("temp_compressed_audio.wav", rate=sample_rate, data=compressed_signal)

    # Exporter le signal compressé en format MP3
    compressed_audio.export("compressed_audio.mp3", format="mp3")

    # Supprimer le fichier WAV temporaire
    os.remove("temp_compressed_audio.wav")

    return compressed_signal

# Fonction pour réduire le bruit du signal fenêtré et afficher les résultats
def reduire_bruit_et_afficher_resultats():
    global signal_fenetree, fig_fenetree, ax_fenetree, canvas_fenetree
    if fig_spectre is not None:
        fig_spectre.clf()
        canvas_spectre.get_tk_widget().destroy()
        # Créer une nouvelle figure pour le signal fenêtré bruité
    if fig_fenetree is not None:
        fig_fenetree.clf()
        canvas_fenetree.get_tk_widget().destroy()

    # Sauvegarder le signal bruité avant réduction
    signal_bruite = np.copy(signal_fenetree)

    # Choisissez une taille de filtre adaptée à votre cas
    my_filter_size = 10

    # Appliquer la réduction de bruit au signal fenêtré
    signal_fenetree_filtre = wiener(signal_fenetree, my_filter_size)  # Assurez-vous d'ajuster my_filter_size

    # Créer une nouvelle figure pour le signal fenêtré filtré
    fig_fenetree, ax_fenetree = plt.subplots(figsize=(5, 2))
    fig_fenetree.patch.set_facecolor('#F0F0F0')  # Couleur de fond du graphique
    fig_fenetree.canvas.draw()
    canvas_fenetree = FigureCanvasTkAgg(fig_fenetree, master=root)
    canvas_widget_fenetree = canvas_fenetree.get_tk_widget()
    canvas_widget_fenetree.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Afficher le signal bruité avant réduction
    ax_fenetree.plot(temps[:len(signal_bruite)], signal_bruite, label="Signal bruité")

    # Afficher le signal fenêtré filtré dans la nouvelle figure
    ax_fenetree.plot(temps[:len(signal_fenetree_filtre)], signal_fenetree_filtre, label="Signal filtré")
    ax_fenetree.set_xlabel('Temps (s)')
    ax_fenetree.set_ylabel('Amplitude')
    ax_fenetree.set_title('Signal musical filtré', fontsize=10, fontweight='bold', fontstyle='italic', color='blue')
    ax_fenetree.legend()
    canvas_fenetree.draw()


def apply_window(signal_segment):
    window = np.hamming(len(signal_segment))
    return signal_segment * window

# Fonction pour mettre à jour le graphique après la compression
def update_after_compression(compressed_signal):
    global ax, canvas

    # Mettre à jour le graphique dans le domaine temporel avec le signal compressé
    ax.clear()
    ax.plot(temps, compressed_signal)
    ax.set_xlabel('Temps (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Signal musical compressé', fontsize=10, fontweight='bold', fontstyle='italic', color='blue')
    ax.legend(["Signal compressé"])
    canvas.draw()
def apply_window(signal_segment):
    window = np.hamming(len(signal_segment))
    return signal_segment * window

# Fonction pour arrêter l'acquisition
def stop_acquisition():
    global acquisition_en_cours, afficher_signal
    acquisition_en_cours = False
    afficher_signal = False
    update_graph()

# Fonction pour mettre à jour le graphique
def update_graph():
    global index, acquisition_en_cours, signal_fenetree, afficher_signal, signal_audio_a_afficher
    ax.clear()
    if acquisition_en_cours:
        signal_sous_echantillonne = apply_window(signal_audio_a_afficher[index:index + 2000])
        temps_sous_echantillonne = temps[index:index + 2000]
        ax.plot(temps_sous_echantillonne, signal_sous_echantillonne)
        ax.set_xlabel('Temps (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Signal musical', fontsize=10, fontweight='bold', fontstyle='italic', color='blue')
        ax.legend(["Signal"])
        canvas.draw()
        index += 2000
        root.after(300, update_graph)
    elif signal_fenetree is not None and afficher_signal:
        ax.plot(temps[:len(signal_fenetree)], signal_fenetree, label="Signal Fenêtré", color='blue')

        # Ajouter une ligne pour tracer le signal bruité en rouge
        ax.plot(temps[:len(signal_audio_a_afficher)], signal_audio_a_afficher, label="Signal Bruité", color='red')

        # Ajouter une ligne pour tracer le signal compressé en vert
        ax.plot(temps[:len(signal_audio_a_afficher)], compress_signal(signal_audio_a_afficher, 0.5), label="Signal Compressé", color='green')

        ax.set_xlabel('Temps (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Signal musical (Fenêtré, Bruité, Compressé)', fontsize=11, fontweight='bold', fontstyle='italic',
                     color='blue')
        ax.legend()
        canvas.draw()


# Variable pour stocker la figure du signal fenêtré
fig_fenetree = None
ax_fenetree = None
canvas_fenetree = None

# Variable pour stocker la figure du spectre
fig_spectre = None
ax_spectre = None
canvas_spectre = None

# Nouvelle fonction pour afficher le spectre sur tout le signal
def afficher_spectre_sur_signal_complet():
    global signal_audio_a_afficher, fig_spectre, ax_spectre, canvas_spectre

    # Effacer le contenu de la figure existante
    if fig_spectre is not None:
        fig_spectre.clf()
        canvas_spectre.get_tk_widget().destroy()
        # Créer une nouvelle figure pour le signal fenêtré filtré
    if fig_fenetree is not None:
            fig_fenetree.clf()
            canvas_fenetree.get_tk_widget().destroy()
    if fig_spectrogram is not None:
        fig_spectrogram.clf()
        canvas_spectrogram.get_tk_widget().destroy()
    # Créer une nouvelle figure pour le spectre
    fig_spectre, ax_spectre = plt.subplots(figsize=(4, 2))
    fig_spectre.patch.set_facecolor('#F0F0F0')  # Couleur de fond du graphique
    fig_spectre.canvas.draw()
    canvas_spectre = FigureCanvasTkAgg(fig_spectre, master=root)
    canvas_widget_spectre = canvas_spectre.get_tk_widget()
    canvas_widget_spectre.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Assurez-vous que signal_audio_a_afficher est correctement défini
    if signal_audio_a_afficher is not None:
        # Appliquer la transformée de Fourier sur le signal complet
        spectre_complet = fft(signal_audio_a_afficher)

        # Calculer les fréquences correspondantes
        frequences_complet = np.fft.fftfreq(len(spectre_complet), d=1/sample_rate)

        # Afficher le spectre d'amplitude du signal complet
        ax_spectre.plot(frequences_complet, np.abs(spectre_complet))
        ax_spectre.set_xlabel('Fréquence (Hz)')
        ax_spectre.set_ylabel('Amplitude du spectre')
        ax_spectre.set_title('Spectre d\'amplitude du signal complet', fontsize=10, fontweight='bold', fontstyle='italic', color='purple')
        ax_spectre.legend(["Spectre complet"])
        canvas_spectre.draw()


# Fonction pour afficher le spectre d'amplitude après le fenêtrage
def afficher_spectre_apres_fenetrage():
    global signal_fenetree, fig_spectre, ax_spectre, canvas_spectre

    # Assurez-vous que signal_fenetree est correctement défini
    if signal_fenetree is not None:
        # Effacer le contenu de la figure existante
        if fig_spectre is not None:
            fig_spectre.clf()
            canvas_spectre.get_tk_widget().destroy()
        # Créer une nouvelle figure pour le signal fenêtré filtré
        if fig_fenetree is not None:
            fig_fenetree.clf()
            canvas_fenetree.get_tk_widget().destroy()
        if fig_spectrogram is not None:
            fig_spectrogram.clf()
            canvas_spectrogram.get_tk_widget().destroy()
        # Créer une nouvelle figure pour le spectre
        fig_spectre, ax_spectre = plt.subplots(figsize=(4, 2))
        fig_spectre.patch.set_facecolor('#F0F0F0')  # Couleur de fond du graphique
        fig_spectre.canvas.draw()
        canvas_spectre = FigureCanvasTkAgg(fig_spectre, master=root)
        canvas_widget_spectre = canvas_spectre.get_tk_widget()
        canvas_widget_spectre.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Appliquer la transformée de Fourier sur le signal fenêtré
        spectre_fenetree = fft(signal_fenetree)

        # Calculer les fréquences correspondantes
        frequences_fenetree = np.fft.fftfreq(len(spectre_fenetree), d=1/sample_rate)

        # Afficher le spectre d'amplitude du signal fenêtré
        ax_spectre.plot(frequences_fenetree, np.abs(spectre_fenetree))
        ax_spectre.set_xlabel('Fréquence (Hz)')
        ax_spectre.set_ylabel('Amplitude du spectre')
        ax_spectre.set_title('Spectre d\'amplitude du signal fenêtré', fontsize=10, fontweight='bold', fontstyle='italic', color='purple')
        ax_spectre.legend(["Spectre fenêtré"])
        canvas_spectre.draw()

# Fonction pour démarrer l'acquisition
def start_acquisition():
    global index, acquisition_en_cours,afficher_signal
    index = 0
    acquisition_en_cours = True
    afficher_signal = False
    update_graph()

# Fonction pour gérer l'effet de surbrillance des boutons
def on_enter(event):
    event.widget.config(relief=tk.RAISED)
def on_leave(event):
    event.widget.config(relief=tk.FLAT)

# Bouton de démarrage (vert, gras, italique)
start_button = tk.Button(root, text="Démarrer l'acquisition", command=start_acquisition, font=("Arial", 10, "bold italic"), bg='green')
start_button.pack(side=tk.LEFT)
start_button.bind("<Enter>", on_enter)
start_button.bind("<Leave>", on_leave)
# Ajouter de l'espace entre les boutons
spacer = tk.Label(root, text="")
spacer.pack(side=tk.LEFT)

# Bouton d'arrêt (rouge, gras, italique)
stop_button = tk.Button(root, text="Arrêter l'acquisition", command=stop_acquisition, font=("Arial", 10, "bold italic"), bg='red')
stop_button.pack(side=tk.LEFT)
stop_button.bind("<Enter>", on_enter)
stop_button.bind("<Leave>", on_leave)

# Ajouter de l'espace entre les boutons
spacer = tk.Label(root, text="")
spacer.pack(side=tk.LEFT)

# Bouton Fenêtrer le signal (bleu, gras, italique)
fenetrer_button = tk.Button(root, text="fenetrer_signal", command=fenetrer_signal, font=('Arial', 10, 'bold'),bg='violet', foreground='navy')
fenetrer_button.pack(side=tk.LEFT)
fenetrer_button.bind("<Enter>", on_enter)
fenetrer_button.bind("<Leave>", on_leave)
# Ajouter de l'espace entre les boutons
spacer = tk.Label(root, text="")
spacer.pack(side=tk.LEFT)

def basculer_affichage_spectre():
    afficher_spectre_sur_signal_complet()
    afficher_spectre_apres_fenetrage()

# Bouton Afficher Spectre (violet, gras, italique)
afficher_spectre_button = tk.Button(root, text="Afficher Spectre", command=basculer_affichage_spectre, font=('Arial', 10, 'bold'), bg='#8B4513', foreground='navy')
afficher_spectre_button.pack(side=tk.LEFT)
afficher_spectre_button.bind("<Enter>", on_enter)
afficher_spectre_button.bind("<Leave>", on_leave)
# Ajouter de l'espace entre les boutons
spacer = tk.Label(root, text="")
spacer.pack(side=tk.LEFT)


# Bouton Égaliser (gris, gras, italique)
egaliser_button = tk.Button(root, text="Égaliser", command=apply_equalization, font=('Arial', 10, 'bold'),bg='yellow', foreground='navy')
egaliser_button.pack(side=tk.LEFT)
egaliser_button.bind("<Enter>", on_enter)
egaliser_button.bind("<Leave>", on_leave)

# Ajouter de l'espace entre les boutons
spacer = tk.Label(root, text="")
spacer.pack(side=tk.LEFT)

# Définir la fonction qui sera appelée lors du clic sur le bouton
def on_compress_button_click():
    # Assurez-vous de passer le signal_audio_a_afficher comme argument à la fonction
    compress_signal_mp3(signal_audio_a_afficher)

# Créer le bouton en utilisant la nouvelle fonction définie
compression_mp3_button = tk.Button(root, text="Compress Signal MP3", command=on_compress_button_click, font=('Arial', 10, 'bold'),bg='Royal Blue', foreground='navy')
compression_mp3_button.pack(side=tk.LEFT)
compression_mp3_button.bind("<Enter>", on_enter)
compression_mp3_button.bind("<Leave>", on_leave)
# Ajouter de l'espace entre les boutons
spacer = tk.Label(root, text="")
spacer.pack(side=tk.LEFT)

ajout_bruit_button = tk.Button(root, text="Ajouter du bruit", command=ajout_bruit, font=('Arial', 10, 'bold'),bg='orange', foreground='navy')
ajout_bruit_button.pack(side=tk.LEFT)
ajout_bruit_button.bind("<Enter>", on_enter)
ajout_bruit_button.bind("<Leave>", on_leave)

# Ajouter de l'espace entre les boutons
spacer = tk.Label(root, text="")
spacer.pack(side=tk.LEFT)

# Bouton Réduire le bruit (gris, gras, italique)
reduire_bruit_button = tk.Button(root, text="filtrage wiener", command=reduire_bruit_et_afficher_resultats, font=('Arial', 10, 'bold'), bg='gray', foreground='navy')
reduire_bruit_button.pack(side=tk.LEFT)
reduire_bruit_button.bind("<Enter>", on_enter)
reduire_bruit_button.bind("<Leave>", on_leave)
# Ajouter de l'espace entre les boutons
spacer = tk.Label(root, text="")
spacer.pack(side=tk.LEFT)

# Vous pouvez quitter l'interface graphique en fermant la fenêtre.
root.mainloop()