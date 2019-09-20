/*
Pitch detection over multiple renditions of a given syllable, using FFTW (http://www.fftw.org/).
To compile, install FFTW and link it: g++ -lfftw3 Pitch.cpp
*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <fftw3.h>
#include <math.h>
using namespace std;

#define REAL 0
#define IMAG 1

int N = 512;																						// No. of samples considered
int fs = 32000;																						// Sampling rate
int lag = 200;																						// Lag from syllable onset

class Annotation	{
	public:
		int onset, offset;
		char label;
};

class FFT	{
	public:
		double energy, freq;
};

void readfiles(vector <double> &song, vector <Annotation> &labels)	{
	/* Read song file and labels. */

	ifstream songfile("Combined_songs.txt");
	ifstream labelfile("Combined_labels_tampered.txt");
	
	cout << "Reading song." << endl;
	
	double amp;
	while( songfile >> amp )	{
	    song.push_back(amp);
	} 
	
	string line;
	cout << "Reading labels." << endl;

	while( labelfile >> line )	{

	    stringstream ss(line);
	    string token;
	    Annotation annot;
	    int i=0;
        while(getline(ss, token, ',')){																// Separates an annotation into its tokens: onset, offset, label
             switch(i){
             	case 0:	annot.onset = stoll(token);	break;
             	case 1:	annot.offset = stoll(token);	break;
             	case 2: annot.label = token[0]; break;
             }
             i++;
        }
        labels.push_back(annot);
	}  

	return;
}

vector <FFT> compute_fft(vector <double> &sample, size_t n)	{
	/* Computes FFT using the FFTW library. */

    fftw_complex in[n], out[n];
    vector <FFT> course_fft;

    for(size_t i=0; i<n; i++)	{
    	in[i][REAL] = sample[i];
    	in[i][IMAG] = 0;
    }

    fftw_plan plan = fftw_plan_dft_1d(n, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);	
    fftw_cleanup();
    
    /* Note: FFTs in the FFTW output are stored with the positive frequencies in the first half of the array, and with the negative frequencies in the second half in reverse order. */
    for(size_t i=0; i<n/2; i++)	{														            // Constructs FFT object from FFTW output
    	FFT curr;
    	curr.energy = fabs(out[i][REAL]);														    // Stores energy as absolute values of the real component of FFTW output
    	curr.freq = ((double)(i)*fs)/n;																// Computes frequency in Hz from the index of FFTW output
    	course_fft.push_back(curr);
    }

	return course_fft;																				// Returns the course FFT made of n samples
}

vector <FFT> construct_fft(vector <double> &signal)	{
	/* Builds 10 FFTs for a signal, each with 2 samples lesser, and combines them to obtain a finer FFT. */

	vector <FFT> course_fft;															            // Coarse FFT of N, or N-2, or ... samples
	vector <FFT> fine_fft;																			
	
    // Constructing 10 course FFTS
    for(int m=0; m<20; m+=2)	{
		course_fft = compute_fft(signal, N-m);
        fine_fft.insert(fine_fft.end(), course_fft.begin(), course_fft.end());                      // Collects all the course_ffts
    }

    // Sorts the combined FFT according to the frequency value
	sort(fine_fft.begin(), fine_fft.end(), [](const FFT &a, const FFT &b) { return a.freq < b.freq;});
    
    /*
    ofstream fftFile("fft_file.txt");															    // Uncomment if you wish to write the FFT to a file (Python support provided for plotting)
    for (const FFT &e : fine_fft) fftFile << e.freq << ',' << e.energy << "\n";
     */
    
	return fine_fft;
}

double calculate_pitch(vector <FFT> fft)	{

    // Restricts search range of lowest harmonic to a 30% variation around 900Hz
	vector <FFT>::iterator range_beg = lower_bound(fft.begin(), fft.end(), 600.0, [](const FFT &a, const double v) { return a.freq < v;});
	vector <FFT>::iterator range_end = lower_bound(fft.begin(), fft.end(), 1200.0, [](const FFT &a, const double v) { return a.freq < v;});

	vector <FFT>::iterator max_it = max_element(range_beg, range_end, [](const FFT &a, const FFT &b) { return a.energy < b.energy;});					// Finds FFT energy peak
	size_t max_ind = distance(fft.begin(), max_it);													// Finds corresponding index

	double pitch = max_it->freq;																	// Pitch is the frequency at the peak in the FFT

    /*
	ofstream pitchFile("pitch_file.txt");														    // Uncomment if you wish to write the pitch value to a file (Python support provided for reading)
 	pitchFile << pitch << endl;
    */

	return pitch;
}

void collect_pitches(vector <double> &song, vector <Annotation> &labels)	{
	/* Processes occurences of syllable F. */

	vector <double> pitches;																		// Stores pitch at each rendition of syllable F

	for(vector<Annotation>::iterator it = labels.begin(); it != labels.end(); ++it)	{

		if( it->label == 'f')	{

			vector <double> syllable ( song.begin()+it->onset, song.begin()+it->offset );			// A syllable is the signal between the onset and offset
			if(syllable.size() < lag + N)	continue;
			vector <double> signal ( syllable.begin()+lag, syllable.begin()+lag+N );				// FFT is computed with N samples starting at a lag from syllable onset. 
			vector <FFT> fine_fft ( construct_fft(signal) );										// Computes the FFT
			double pitch = calculate_pitch(fine_fft);												// Computes the pitch
			pitches.push_back(pitch);
		}
	}

	ofstream pitchFile("pitch_file.txt");															// Writes the pitch of each rendition of syllable F into a file, for further analysis using Python.
	for (const double &e : pitches) pitchFile << e << endl;;

	return;
}

int main () {
    
    vector <double> song;
    vector <Annotation> labels;

	readfiles(song, labels);																		// Loads song signal and annotations
	collect_pitches(song, labels);																	// Calculates pitch at each rendition of syllable F

	return 0;
}
