"""
SoundTerminal

SoundTerminal is meant to be fun. It is a sample mixer and sequencer meant to create bangers in your favourite terminal. 
Inspired by the Roland TR-909.
Sample mix rate is configured at 48.1 khz. You may want to change this if most of the samples you're using are
of a different sample rate (such as 48Khz), to avoid the slight loss of quality due to resampling.

"""
import os
import cmd
import sys
from sys import platform
import random
import glob
from configparser import ConfigParser
from soundterminal.sample import Sample
from soundterminal.playback import Output
from soundterminal import params
from soundterminal import playback
from soundterminal import generative
from soundterminal import streaming
from asciistuff import Banner, Lolcat, Quote

class Mixer:
    """
    Mixes a set of ascii-bar tracks using the given sample instruments, into a resulting big sample.
    """
    def __init__(self, patterns, bpm, ticks, instruments):
        for p in patterns:
            bar_length = 0
            for instrument, bars in p.items():
                if instrument not in instruments:
                    raise ValueError("instrument '{:s}' not defined".format(instrument))
                if len(bars) % ticks != 0:
                    raise ValueError("bar length must be multiple of the number of ticks")
                if 0 < bar_length != len(bars):
                    raise ValueError("all bars must be of equal length in the same pattern")
                bar_length = len(bars)
        self.patterns = patterns
        self.instruments = instruments
        self.bpm = bpm
        self.ticks = ticks

    def mix(self, verbose=True):
        """
        Mix all the patterns into a single result sample.
        """
        if not self.patterns:
            if verbose:
                print("No patterns to mix, output is empty.")
            return Sample()
        total_seconds = 0.0
        for p in self.patterns:
            bar = next(iter(p.values()))
            total_seconds += len(bar) * 60.0 / self.bpm / self.ticks
        if verbose:
            print("__________________________________\n")
            print("Mixing {:d} patterns ...".format(len(self.patterns)))
        mixed = Sample().make_32bit()
        for index, timestamp, sample in self.mixed_samples(tracker=False):
            mixed.mix_at(timestamp, sample)
        # chop/extend to get to the precise total duration (in case of silence in the last bars etc)
        missing = total_seconds-mixed.duration
        if missing > 0:
            mixed.add_silence(missing)
        elif missing < 0:
            mixed.clip(0, total_seconds)

        return mixed

    def mix_generator(self):
        """
        Returns a generator that produces samples that are the chronological
        chunks of the final output mix. This avoids having to mix it into one
        big output mix sample.
        """
        if not self.patterns:
            yield Sample()
            return
        total_seconds = 0.0
        for p in self.patterns:
            bar = next(iter(p.values()))
            total_seconds += len(bar) * 60.0 / self.bpm / self.ticks
        mixed_duration = 0.0
        samples = self.mixed_samples()
        # get the first sample
        index, previous_timestamp, sample = next(samples)
        mixed = Sample().make_32bit()
        mixed.mix_at(previous_timestamp, sample)
        # continue mixing the following samples
        for index, timestamp, sample in samples:
            trigger_duration = timestamp-previous_timestamp
            overflow = None
            if mixed.duration < trigger_duration:
                # fill with some silence to reach the next sample position
                mixed.add_silence(trigger_duration - mixed.duration)
            elif mixed.duration > trigger_duration:
                # chop off the sound that extends into the next sample position
                # keep this overflow and mix it later!
                overflow = mixed.split(trigger_duration)
            mixed_duration += mixed.duration
            yield mixed
            mixed = overflow if overflow else Sample().make_32bit()
            mixed.mix(sample)
            previous_timestamp = timestamp
        # output the last remaining sample and extend it to the end of the duration if needed
        timestamp = total_seconds
        trigger_duration = timestamp-previous_timestamp
        if mixed.duration < trigger_duration:
            mixed.add_silence(trigger_duration - mixed.duration)
        elif mixed.duration > trigger_duration:
            mixed.clip(0, trigger_duration)
        mixed_duration += mixed.duration
        yield mixed

    def mixed_triggers(self, tracker):
        """
        Generator for all triggers in chronological sequence.
        Every element is a tuple: (trigger index, time offset (seconds), list of (instrumentname, sample tuples)
        """
        time_per_index = 60.0 / self.bpm / self.ticks
        index = 0
        for pattern_nr, pattern in enumerate(self.patterns, start=1):
            pattern = list(pattern.items())
            num_triggers = len(pattern[0][1])
            for i in range(num_triggers):
                triggers = []
                triggered_instruments = set()
                for instrument, bars in pattern:
                    if bars[i] not in ". ":
                        sample = self.instruments[instrument]
                        triggers.append((instrument, sample))
                        triggered_instruments.add(instrument)
                if triggers:
                    if tracker:
                        triggerdots = ['#' if instr in triggered_instruments else '.' for instr in self.instruments]
                        print("\r{:3d} [{:3d}] ".format(index, pattern_nr), "".join(triggerdots), end="   ", flush=True)
                    yield index, time_per_index*index, triggers
                index += 1

    def mixed_samples(self, tracker=True):
        """
        Generator for all samples-to-mix.
        Every element is a tuple: (trigger index, time offset (seconds), sample)
        """
        mix_cache = {}  # we cache stuff to avoid repeated mixes of the same instruments
        for index, timestamp, triggers in self.mixed_triggers(tracker):
            if len(triggers) > 1:
                # sort the samples to have the longest one as the first
                # this allows us to allocate the target mix buffer efficiently
                triggers = sorted(triggers, key=lambda t: t[1].duration, reverse=True)
                instruments_key = tuple(instrument for instrument, _ in triggers)
                if instruments_key in mix_cache:
                    yield index, timestamp, mix_cache[instruments_key]
                    continue
                # duplicate the longest sample as target mix buffer, then mix the remaining samples into it
                mixed = triggers[0][1].copy()
                for _, sample in triggers[1:]:
                    mixed.mix(sample)
                mixed.lock()
                mix_cache[instruments_key] = mixed   # cache the mixed instruments sample
                yield index, timestamp, mixed
            else:
                # simply yield the unmixed sample from the single trigger
                yield index, timestamp, triggers[0][1]

class Song:
    """
    Represents a set of instruments, patterns and bars that make up a 'track'.
    """
    def __init__(self):
        self.instruments = {}
        self.sample_path = None
        self.bpm = 128
        self.ticks = 4
        self.pattern_sequence = []
        self.patterns = {}
        self.currProject = 'empty'
        self.loaded = False

    def read(self, song_file, discard_unused_instruments=True):
        """Read a song from a saved file."""
        with open(song_file):
            pass    # test for file existence

        bnnr = song_file.split("/")
        print(Banner(bnnr[1]))
        print('_____________________________________________\n')
        print("Project: " + song_file + '\n')
        cp = ConfigParser()
        cp.read(song_file)
        self.sample_path = cp["paths"]["samples"]
        self.read_samples(cp["samples"], self.sample_path)
        if "song" in cp:
            self.bpm = cp["song"].getint("bpm")
            self.ticks = cp["song"].getint("ticks")
            self.read_patterns(cp, cp["song"]["patterns"].split())
            print("BPM: " + str(self.bpm))
            print("Ticks: " + str(self.ticks))
        print("Loaded: {:d} samples and {:d} patterns".format(len(self.instruments), len(self.patterns)))
    
        unused_instruments = self.instruments.keys()
        for pattern_name in self.pattern_sequence:
            unused_instruments -= self.patterns[pattern_name].keys()
        if unused_instruments and discard_unused_instruments:
            for instrument in list(unused_instruments):
                del self.instruments[instrument]
            print("Info: Unused samples have been unloaded to save memory.")
            print("Unloaded samples:", ", ".join(sorted(unused_instruments)))
            print("\r                  ")

    def read_samples(self, instruments, samples_path):
        """Reads the sample files for the instruments."""
        self.instruments = {}
        for name, file in sorted(instruments.items()):
            self.instruments[name] = Sample(wave_file=os.path.join(samples_path, file)).normalize().make_32bit(scale_amplitude=False).lock()

    def read_patterns(self, songdef, names):
        """Reads and parses the pattern specs from the song."""
        self.pattern_sequence = []
        self.patterns = {}
        for name in names:
            if "pattern."+name not in songdef:
                raise ValueError("pattern definition not found: "+name)
            bar_length = 0
            self.patterns[name] = {}
            for instrument, bars in songdef["pattern."+name].items():
                if instrument not in self.instruments:
                    raise ValueError("instrument '{instr:s}' not defined (pattern: {pattern:s})".format(instr=instrument, pattern=name))
                bars = bars.replace(' ', '')
                if len(bars) % self.ticks != 0:
                    raise ValueError("all patterns must be multiple of song ticks (pattern: {pattern:s}.{instr:s})"
                                     .format(pattern=name, instr=instrument))
                self.patterns[name][instrument] = bars
                if 0 < bar_length != len(bars):
                    raise ValueError("all bars must be of equal length in the same pattern (pattern: {pattern:s}.{instr:s})"
                                     .format(pattern=name, instr=instrument))
                bar_length = len(bars)
            self.pattern_sequence.append(name)

    def write(self, output_filename):
        """Save the song definitions to an output file."""
        import collections
        cp = ConfigParser(dict_type=collections.OrderedDict)
        cp["paths"] = {"samples": self.sample_path}
        cp["song"] = {"bpm": self.bpm, "ticks": self.ticks, "patterns": " ".join(self.pattern_sequence)}
        cp["samples"] = {}
        for name, sample in sorted(self.instruments.items()):
            cp["samples"][name] = os.path.basename(sample.filename)
           
        for name, pattern in sorted(self.patterns.items()):
            # Note: the layout of the patterns is not optimized for human viewing. You may want to edit it afterwards.
            cp["pattern."+name] = collections.OrderedDict(sorted(pattern.items()))
        with open(output_filename, 'w') as f:
            cp.write(f)
        print('_____________________________________________\n')
        #print("Project file updated'{:s}'.".format(output_filename))
        print("Track updated")
        print('_____________________________________________\n')

    def mix(self, output_filename):
        """
        Mix the song into a resulting mix sample.
        """
        if not self.pattern_sequence:
            raise ValueError("There's nothing to be mixed; no song loaded or song has no patterns.")
        patterns = [self.patterns[name] for name in self.pattern_sequence]
        mixer = Mixer(patterns, self.bpm, self.ticks, self.instruments)
        result = mixer.mix()
        result.make_16bit()
        result.write_wav(output_filename)
        print("Duration {:.2f} seconds".format(result.duration))
        return result

    def mixed_triggers(self):
        """
        Generator that produces all the instrument triggers needed to mix/stream the song.
        Shortcut for Mixer.mixed_triggers, see there for more details.
        """
        patterns = [self.patterns[name] for name in self.pattern_sequence]
        mixer = Mixer(patterns, self.bpm, self.ticks, self.instruments)
        return mixer.mixed_triggers(False)

    def mix_generator(self):
        """
        Generator that produces samples that together form the mixed song.
        Shortcut for Mixer.mix_generator(), see there for more details.
        """
        patterns = [self.patterns[name] for name in self.pattern_sequence]
        mixer = Mixer(patterns, self.bpm, self.ticks, self.instruments)
        return mixer.mix_generator()

class Repl(cmd.Cmd):
    """
    Interactive command line interface to load/record/save and play samples, patterns and whole tracks.
    """
    def __init__(self, discard_unused_instruments=False):
        self.song = Song()
        self.discard_unused_instruments = discard_unused_instruments
        self.out = Output(mixing="sequential", queue_size=1)
        super(Repl, self).__init__()

    def do_exit(self, args):
        """
        Quit the session.
        [args] [none]
        """
        if (self.song.loaded == True):
            self.do_save(self.song.currProject) 
        print("\nSee ya later alligator!", args)
        self.out.close()
        return True
    
    def do_cls(self, args):
        """
        Clear screen command
        """
        if platform == 'win32':
            os.system('cls') 
        else:
            os.system('clear')  

    def do_info(self, args):
        """
        View/set options: 

        [args] [none]
        [args] [device] [0]
        [args] [antipop] [False]
        """
        argArry = args.split()
        if len(argArry) < 2 or len(argArry) > 2:
            print("___________________________\n")
            print("Sample rate: " + str(params.norm_samplerate))
            print("Audio device: " + str(playback.default_audio_device))
            print("Anti-pop: " + str(params.auto_sample_pop_prevention))
            print("Volume: " + str(params.default_audio_volume))
            print("___________________________\n")
            print(
            """Sections & commands:

                [Global] help, info
                [Track] tracks, save, ticks, samples, load, exit, bpm, export,
                [Sequence] seq
                [Pattern] new, playp, print, copy, edit, del
                [Instruments] samples, plays, add

            Made with <3 by github/Omodaka9375
            """)
        else:
            comm, val = argArry
            # if(comm == 'rate'):
            #     if int(val) != 48000 or val != 44100:
            #         print('Invalid sample rate choose: 44100 or 48000')
            #         return
            #     params.norm_samplerate = int(val) 
            if(comm == 'device'):
                playback.default_audio_device = int(val)
            # if(comm == 'volume'):
            #     params.default_audio_volume = int(val)
            if(comm == 'antipop'):
                params.auto_sample_pop_prevention = bool(val)

    def do_bpm(self, bpm):
        """
        Set the playback BPM (such as 174 for some drum'n'bass).
        [args] [none] - show bpm for current track
        [args] [bpm] - set bpm for current track
        """
        if bpm:
            try:
                self.song.bpm = int(bpm)
                print("Bpm " + str(self.song.bpm))
            except ValueError as x:
                print("ERROR:", x)
        else:
            print("Bpm " + str(self.song.bpm))

    def do_ticks(self, ticks):
        """
        Set the number of pattern ticks per beat (usually 4 or 8).
        [args] [none] - show ticks value for current track
        [args] [ticks] - set ticks for the track
        """
        if not ticks:
            print("Ticks: " + str(self.song.ticks))
        else:
            try:
                self.song.ticks = int(ticks)
                print("Ticks: " + str(self.song.ticks))
            except ValueError as x:
                print("ERROR:", x)

    def do_samples(self, args):
        """
        Show loaded samples.
        [args] [none]
        """
        print("Samples:")
        print(",  ".join(self.song.instruments))

    def do_print(self, args):
        """
        Show all loaded patterns
        [args] [none]
        """
        patterns = args.split()
        if len(patterns) <= 0:
            print("\nPatterns:\n")
            for name, pattern in sorted(self.song.patterns.items()):
                self.print_pattern(name, pattern)
        else: 
            for pattn in set(patterns):
                self.print_pattern(pattn, self.song.patterns[pattn])

    def do_del(self,args):
        """
        Delete pattern by name.
        [args] [patternname]
        """
        patterns = args.split()

        if len(patterns) <= 0:
            print("Please provide at least one pattern name to delete")
            return
        for pattn in set(patterns):
            del self.song.patterns[pattn]
            if pattn in self.song.pattern_sequence:
                self.song.pattern_sequence.remove(pattn)
        print("Done.\n")
    
    def do_new(self,args):
        """
        Add new empty pattern by name.
        [args] [patternname]
        """
        patterns = args.split()

        if len(patterns) <= 0:
            print("Please provide pattern name")
            return
        for pattn in set(patterns):
            self.song.patterns[pattn] = {}
        print("Done.\n")

    def do_add(self, args):
        """
        Add sample to a project by name.
        [args] [shortname] [samplefilename.wav]
        Note: All samples need to be inside /samples folder
        """
        namess = args.split()
        if(len(namess) != 2):
            print('Usage: adds shortname samplename.wav')
            return
        short, sampfilename = namess
        self.song.instruments[short] = Sample(wave_file=os.path.join(self.song.sample_path, sampfilename)).normalize().make_32bit(scale_amplitude=False).lock()

    def print_pattern(self, name, pattern):
        print(name)
        for instrument, bars in pattern.items():
            print("   {:>15s} = {:s}".format(instrument, bars))

    def do_playp(self, names):
        """
        Play the given pattern(s).
        [args] [pat1] [pat1]
        """
        names = names.split()
        for name in sorted(set(names)):
            try:
                pat = self.song.patterns[name]
                self.print_pattern(name, pat)
            except KeyError:
                print("No such pattern '{:s}'".format(name))
                print("Try loading a track first: load empty or load track4")
                return
        patterns = [self.song.patterns[name] for name in names]
        try:
            m = Mixer(patterns, self.song.bpm, self.song.ticks, self.song.instruments)
            result = m.mix(verbose=len(patterns) > 1).make_16bit()
            self.out.play_sample(result)
            self.out.wait_all_played()
        except ValueError as x:
            print("ERROR:", x)

    def do_plays(self, args):
        """
        Play a single sample by name, add a bar (xx..x.. etc) to play it in a bar.
        [args] [samplename] - optional [bar]
        """
        if ' ' in args:
            instrument, pattern = args.split(maxsplit=1)
            pattern = pattern.replace(' ', '')
        else:
            instrument = args
            pattern = None
        instrument = instrument.strip()
        try:
            sample = self.song.instruments[instrument]
        except KeyError:
            print("Unknown sample.")
            print('Usage: Play a single sample by name, add a bar (xx..x.. etc) to play it in a bar')
            return
        if pattern:
            self.play_single_bar(sample, pattern)
        else:
            sample = sample.copy().make_16bit()
            self.out.play_sample(sample)
            self.out.wait_all_played()

    def play_single_bar(self, sample, pattern):
        try:
            m = Mixer([{"sample": pattern}], self.song.bpm, self.song.ticks, {"sample": sample})
            result = m.mix(verbose=False).make_16bit()
            self.out.play_sample(result)
            self.out.wait_all_played()
        except ValueError as x:
            print("ERROR:", x)

    def do_play(self, args):
        """
        Play all patterns of the track
        [args] [none]
        """
        if not self.song.pattern_sequence:
            print("Nothing to be mixed")
            return
        output = "_temp_mix.wav"
        self.song.mix(output)
        mix = Sample(wave_file=output)
        print("Playing " + self.song.currProject + " track ...")
        self.do_seq('')
        self.out.play_sample(mix)
        os.remove(output)

    def do_export(self, args):
        """
        Mix all patterns to an output file. You need to provde a filename argument.
        This is the fastest and most efficient way of generating the output mix.
        [args] [filename]
        """
        if not self.song.pattern_sequence:
            print("Nothing to be mixed. Have any patterns loaded?")
            return
        if args:
            filename = args.strip()
            print("________________________________________________")
            print("\nMixing and streaming to file 'outputs/{0}.wav'".format(filename))
            print("________________________________________________")
            self.out.stream_to_file('output/'+ filename + '.wav', self.song.mix_generator())
            print("\r                                               ")
            return
        else:
            print(" - must add filename")
            return

    def do_edit(self, args):
        """
        Add (or overwrite) a new sample (instrument) bar in a pattern.
        [args] [pattern name] [sample]  - optional [bar(s) or gen].
        Omit bars to remove the sample from the pattern.
        If instead of bars you enter [gen] SoundTerminal will generate it for you
        If a pattern with that name doesn't exist yet it will be added.
        """   
        args = args.split(maxsplit=2)
        if len(args) not in (2, 3):
            print("Wrong arguments. Use: patternname sample bar(s)")
            return
        if len(args) == 2:
            args.append(None)   # no bars
        pattern_name, instrument, bars = args
        if instrument not in self.song.instruments:
            print("Unknown sample '{:s}'.".format(instrument))
            return
        if pattern_name not in self.song.patterns:
            self.song.patterns[pattern_name] = {}
            self.song.pattern_sequence.append(pattern_name)

        if bars:
            if bars == 'gen':
                genbars = random.choice(generative.patterns)
                genbars = genbars.replace(' ', '')
                if len(genbars) % self.song.ticks != 0:
                    print("Bar length must be multiple of the number of steps.")
                    return
                self.song.patterns[pattern_name][instrument] = genbars
            else:
                bars = bars.replace(' ', '')
                if len(bars) % self.song.ticks != 0:
                    print("Bar length must be multiple of the number of steps.")
                    return
                self.song.patterns[pattern_name][instrument] = bars
        else:
            if instrument in self.song.patterns[pattern_name]:
                del self.song.patterns[pattern_name][instrument]

        if pattern_name in self.song.patterns:
            if not self.song.patterns[pattern_name]:
                del self.song.patterns[pattern_name]
                print("Pattern was empty and has been removed.")
            else:
                self.print_pattern(pattern_name, self.song.patterns[pattern_name])

    def do_copy(self,args):
        """
        Copy one pattern to another with all it's bars and samples.
        [args] [pattername] [newpatternname]
        """
        argsy = args.split()
        if len(argsy) != 2:
            print("Wrong arguments. Use: copy oldpatternname newpatternname")
            return
        else:
            #copy content from one pattern to another newly created
            oldpattern, newpattern = argsy
            self.song.patterns[newpattern] = self.song.patterns[oldpattern].copy()
            # add new pattern to sequence
            self.song.pattern_sequence.append(newpattern)
            print('__________________________________\n')
            print("Patterns:")
            print('__________________________________\n')
            for name, pattern in sorted(self.song.patterns.items()):
                self.print_pattern(name, pattern)

    def do_seq(self, names):
        """
        Print the sequence of patterns that form the current track,
        or if you give a list of pattern names: use that as the new sequence.
        [args] [none] - show current sequence
        [args] [pat1] [pat1] [pat2] [pat2a] - rearrange patterns inside of sequence
        """
        if not names:
            print('_____________________________________________\n')
            print("Sequence: " + "  ".join(self.song.pattern_sequence))
            print('_____________________________________________\n')
            return
        names = names.split()
        for name in names:
            if name not in self.song.patterns:
                print("Unknown pattern '{:s}'.".format(name))
                return
        self.song.pattern_sequence = names

    def do_load(self, filename):
        """
        Load a new track file by name, or no arguments for empty starter project.
        [args] [none] - load empty starter project
        [args] [projectname] - load specific project/track by name
        """
        song = Song()
        try:
            if not filename:
                filename = 'empty'
            song.read('projects/' + filename + '.st', self.discard_unused_instruments)
            self.song = song
            self.song.currProject = filename
            self.song.loaded = True
            print('_____________________________________________\n')
            print("Patterns:")
            print('_____________________________________________\n')
            for name, pattern in sorted(self.song.patterns.items()):
                self.print_pattern(name, pattern)
            
            self.do_seq('')
        except IOError as x:
            print("ERROR:", x)
            return

    def do_save(self, filename):
        """
        Save current track to file, or pass a new name to save as new track.
        [args] [none] - save current project
        [args] [trackname] - save as different track file
        """
        if not filename:
            filename = self.song.currProject
        if not filename.endswith(".st"):
            filename += ".st"
        if os.path.exists('projects/' + filename):
            if input("File '{:s}' exists. Overwrite y/n? ".format(filename)) not in ('y', 'yes'):
                return
        self.song.write('projects/' + filename)

    def do_tracks(self,args):
        """
        Show all projects tracks
        [args] [none]
        """
        tracklist = glob.glob("projects/*.st")
        print('__________________________________\n')
        for item in tracklist:
            print(item.replace('projects', '').replace('.st',''))
        print('___________________________________\n')

def main(track_file=None):
        if platform == 'win32':
            os.system('cls') 
        else:
            os.system('clear')
        print(Lolcat(Banner("SoundTerminal")))
        discard_unused = False #not interactive
        print('_____________________________________________\n')
        print(Quote("Shut up and take me to the disco!", "- you, probably"))
        repl = Repl(discard_unused_instruments=discard_unused)
        if track_file != None:
            repl.do_load(track_file)

        print('_____________________________________________\n')
        repl.cmdloop("Welcome to SoundTerminal!\n\nType 'help' to get help with commands or 'load' to load starter project:\n")

if __name__ == "__main__":
    if len(sys.argv) > 2:
        print("Usage:")
        print("No arguments needed, or just pass a track name to load it immediately")
    elif len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        main()