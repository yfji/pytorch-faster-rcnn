import threading
import numpy as np

class DataLoader(object):
    def __init__(self, data_loader, shuffle=True, batch_size=1, num_workers=1):
        self.data_loader=data_loader
        self.shuffle=shuffle
        self.batch_size=batch_size
        self.num_workers=num_workers
        self._shuffle()
    
    def __iter__(self):
        return self

    def _shuffle(self):
        self.num_images=self.data_loader.__len__()
        if self.shuffle:
            self.permute_inds=np.random.permutation(np.arange(self.num_images))
        else:
            self.permute_inds=np.arange(self.num_images)
        self.cur_index=0

    def _load(self, roidbs, inds):
        for index in inds:
            roidb=self.data_loader.__getitem__(index)
            if len(roidb)>0:
                roidbs.append(roidb)

    def _prepare_roidb(self, roidbs):
        if len(roidbs)>0 and len(roidbs)<self.batch_size:
            top_n=len(roidbs)
            print('Pad roidbs with previous elements. From {} to {}'.format(top_n, self.batch_size))
            m=self.batch_size//len(roidbs)-1
            n=self.batch_size%len(roidbs)
            for _ in range(m):
                roidbs.extend(roidbs[:top_n])
            if n>0:
                roidbs.extend(roidbs[:n])
            assert len(roidbs)==self.batch_size, 'roidbs length is not valid: {}/{}'.format(len(roidbs), self.batch_size)

    def __next__(self):
        roidbs=[]
        threads=[]
        
        num_threads=min(self.batch_size, self.num_workers)
        roidbs_th=[[] for _ in range(num_threads)]

        if self.num_workers>self.batch_size:
            starts=np.arange(self.batch_size)
            ends=starts+1
        else:
            size_per_thread=self.batch_size//self.num_workers
            starts=size_per_thread*np.arange(self.num_workers)
            ends=size_per_thread+starts
            ends[-1]=max(ends[-1], self.batch_size)

        for i in range(num_threads):
            inds=np.arange(starts[i], ends[i])
            inds=self.permute_inds[self.cur_index+inds]
            t=threading.Thread(target=self._load, args=(roidbs_th[i], inds))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for db in roidbs_th:
            roidbs+=db

#        print('roidbs length: {}'.format(len(roidbs)))
        self.cur_index+=self.batch_size
        if self.cur_index>=self.num_images:
            self._shuffle()
            raise StopIteration
        else:
            self._prepare_roidb(roidbs)
            return roidbs