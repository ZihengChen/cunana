
# @author Ziheng Chen
# @email zihengchen2015@u.northwestern.edu
# @create date 2020-06-16 17:01:14
# @modify date 2020-06-16 17:01:14


from Utility import *
from ExampleEP import *

class Analyzer():
    def __init__(self):
        self.ep = ExampleEP()

        
    def process_infile(self, infile, nev=None):

        eventMask = self.ep.get_mask_for_events_in_file(infile)
        f = root.TFile(infile)
        tree = f.Get("Events")
        if not nev:
            nev = tree.GetEntriesFast()
        eventMask = eventMask[:nev]


        dfs = [list() for _ in range(self.ep.nchannels)]
        # event loop
        for iev in tqdm(range(nev)):
            if not eventMask[iev]:
                continue
            tree.GetEntry(iev)
            self.ep.set_event(tree)
            self.ep.process_event()
            # save selected entry in corresponding list
            if self.ep.out.Channel>=0: 
                dfs[self.ep.out.Channel].append(self.ep.out.copy())
            self.ep.clear_event()


        # convert list of series to df
        dfs = [ pd.DataFrame.from_records(c) for c in dfs]
        
        return dfs


    def process_infiles(self, infiles, outfile):
        if not type(infiles) == list:
            infiles = [infiles]

        if(os.path.exists(outfile)):
            os.remove(outfile)

        # open hdf for storage
        hdf = pd.HDFStore(outfile)
        # infile loop
        for infile in infiles:
            
            time_start = time.time()
            dfs = self.process_infile(infile)
            time_end = time.time()
            print("time {:6.1f}, {} -> {}".format( time_end - time_start, infile, outfile))
            # store dfs in hdf outfile using append mode
            for i in range(self.ep.nchannels):
                hdf.append(self.ep.channelsNames[i], dfs[i], format='t',  data_columns=True)
        
        # close hdf file
        hdf.close()


