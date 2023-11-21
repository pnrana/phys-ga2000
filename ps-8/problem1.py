#Newman 7.3
#Fourier Transforms of musical instruments

import numpy as np
import matplotlib.pyplot as plt

def main():
    #loading waveforms
    piano = np.loadtxt("piano.txt", float)
    trumpet = np.loadtxt("trumpet.txt", float)

    #Plotting both waveforms
    plt.figure(figsize=(8, 8))
    fig = plt.subplot(2, 1, 1)
    plt.xlabel("Time")
    plt.ylabel("Pressure")
    plt.plot(piano)
    plt.title("Piano")

    fig = plt.subplot(2, 1, 2)
    plt.plot(trumpet)
    plt.title("Trumpet",pad=2)
    plt.xlabel("Time")
    plt.ylabel("Pressure")

    plt.subplots_adjust(hspace=0.3)
    plt.suptitle("Waveforms of Piano and Trumpet",fontsize=16)
    #plt.savefig("ps8_1_pt.png",dpi=900)
    plt.show()
    plt.clf()

    #Fourier Transform
    piano_fft = np.fft.fft(piano)
    trumpet_fft = np.fft.fft(trumpet)
    plt.plot(np.abs(piano_fft[:10000]))
    plt.title("Piano FFT")
    plt.xlabel("k")
    plt.ylabel("$|c_k|$")
    #plt.savefig("ps8_1_pianofft.png",dpi=900)
    plt.show()
    plt.clf()

    plt.plot(np.abs(trumpet_fft[:10000]))
    plt.title("Trumpet FFT")
    plt.xlabel("k")
    plt.ylabel("$|c_k|$")
    plt.show()
    #plt.savefig("ps8_1_trumpetfft.png",dpi=900)
    plt.clf()

    #finding the note

    freq_interval = 44100/len(piano_fft)

    #finding magnitudes
    piano_mag = np.abs(piano_fft)
    trumpet_mag = np.abs(trumpet_fft)

    peaks_indices = [np.argmax(piano_mag),np.argmax(trumpet_mag)]
    print("Piano peak frequency: ", (peaks_indices[0]*freq_interval))
    print("Trumpet peak frequency: ", (peaks_indices[1]*freq_interval))

    #find all peak frequencies

    #takes average of close frequencies and returns average frequencies at unique ranges
    def unique_peaks(mags,freq_interval,th,gdiff):
        #stores similar frequencies that will be evaluated
        compare = []
        #stores indices to refer to original magnitudes array
        indices = []
        #stores unique frequencies
        peaks =[]

        for i in range(1,len(mags)):
            if mags[i] > th:
                compare.append(i*freq_interval)
                indices.append(i)

                diff = np.diff(compare)
                #if the frequency just added is not close to the previous element,
                # take frequency using index with greatest magnitude of all elements we have so far
                if (len(diff)>0 and diff[-1] > gdiff):
                    filtered_mags = mags[indices]
                    max_val = np.max(filtered_mags)
                    max_indice = np.where(mags==max_val)[0][0]

                    #update peaks with the frequency with the greatest magnitude among the similar frequencies
                    peaks.append(max_indice*freq_interval)
                    indices= []

                    #reset averaging list with the newsest element
                    compare = [i*freq_interval]

        return peaks

    thp = 5e6
    tht = 5e5

    print("\nPrinting all frequencies: ")
    print("Piano peak frequencies: ", np.sort(unique_peaks(piano_mag,freq_interval,thp,10)))
    print("Trumpet peak frequencies: ", np.sort(unique_peaks(trumpet_mag,freq_interval,tht,10)))













if __name__ == '__main__':
    main()



