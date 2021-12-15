import numpy  as np 
import torch
import torch.utils.data as data

class DataContainer:
    
    def __repr__(self):
         return "DataContainer"
    
    def __init__(
        self, filename, ntrain, nvalid, batch_size=1, valid_batch_size=1, 
        seed=None, dtype=torch.float32):
        
        # Read in data
        dictionary = np.load(filename)
        
        # Number of atoms
        if 'N' in dictionary: 
            self.N = lambda idx:torch.tensor(dictionary['N'][idx], dtype=dtype)
            self.ndata = dictionary['N'].shape[0]
        else:
            raise IOError(
                'The information for the Atom Numbers N are essential')
        
        # Atomic numbers/nuclear charges
        if 'Z' in dictionary: 
            self.Z = lambda idx: torch.tensor(dictionary['Z'][idx], dtype=dtype)
        else:
            raise IOError(
                'The information about the Atomic numbers Z are essential')      
        # Positions (cartesian coordinates)
        if 'R' in dictionary:     
            self.R = lambda idx: torch.tensor(dictionary['R'][idx], dtype=dtype).requires_grad_(True)
        else:
            raise IOError(
                'The information about the Atomic positions R are essential')       
        # Reference energy
        if 'E' in dictionary:
            self.E = lambda idx: torch.tensor(dictionary['E'][idx], dtype=dtype).requires_grad_(True)
            self.include_E = True
        else:
            self.E = lambda idx: None
            self.include_E = False
            
        # Reference atomic energies
        if 'Ea' in dictionary:
            self.Ea = lambda idx: torch.tensor(dictionary['Ea'][idx], dtype=dtype)
            self.include_Ea = True
        else:
            self.Ea = lambda idx: None
            self.include_Ea = False
        
        # Reference forces
        if 'F' in dictionary:
            self.F = lambda idx: torch.tensor(dictionary['F'][idx], dtype=dtype).requires_grad_(True)
            self.include_F = True
        else:
            self.F = lambda idx: None
            self.include_F = False
        
        # Reference total charge
        if 'Q' in dictionary: 
            self.Q = lambda idx: torch.tensor(dictionary['Q'][idx], dtype=dtype).requires_grad_(True)
            self.include_Q = True
        else:
            self.Q = lambda idx: None
            self.include_Q = False
        
        # Reference atomic charges
        if 'Qa' in dictionary: 
            self.Qa = lambda idx: torch.tensor(dictionary['Qa'][idx], dtype=dtype)
            self.include_Qa = True
        else:
            self.Qa = lambda idx: None
            self.include_Qa = False
        
        # Reference dipole moment vector
        if 'D' in dictionary: 
            self.D = lambda idx: torch.tensor(dictionary['D'][idx], dtype=dtype).requires_grad_(True)
            self.include_D = True
        else:
            self.D = lambda idx: None
            self.include_D = False
        
        # Assign parameters
        #self._ndata = self._N.shape[0]
        self.ntrain = ntrain
        self.nvalid = nvalid
        self.ntest = self.ndata - self.ntrain - self.nvalid
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size
        self.dtype = dtype
        
        # Random state parameter for reproducible random operations
        self.random_state = np.random.RandomState(seed=seed)
        
        # Create shuffled list of indices
        idx = self.random_state.permutation(np.arange(self.ndata))
        
        # Store indices of training, validation and test data
        self.idx_train = idx[0:self.ntrain]
        self.idx_valid = idx[self.ntrain:self.ntrain+self.nvalid]
        self.idx_test  = idx[self.ntrain+self.nvalid:]
        
        # Initialize mean/stdev of properties
        self._EperA_mean  = None
        self._EperA_stdev = None
        self._FperA_mean  = None
        self._FperA_stdev = None
        self._DperA_mean  = None
        self._DperA_stdev = None
        
        # Create DataSet for training and valid data
        train_data_all = [
            self.N(self.idx_train), self.Z(self.idx_train), 
            self.R(self.idx_train), self.E(self.idx_train),
            self.Ea(self.idx_train), self.F(self.idx_train), 
            self.Q(self.idx_train), self.Qa(self.idx_train), 
            self.D(self.idx_train)]
        
        self.train_data = [ ]
        for i in train_data_all:
            if i is not None:
                self.train_data.append(i)
            else:
                self.train_data.append(torch.zeros(self.ntrain))
        

        valid_data_all =[ 
            self.N(self.idx_valid), self.Z(self.idx_valid), 
            self.R(self.idx_valid), self.E(self.idx_valid),
            self.Ea(self.idx_valid), self.F(self.idx_valid), 
            self.Q(self.idx_valid), self.Qa(self.idx_valid), 
            self.D(self.idx_valid)]
        
        self.valid_data = [ ]
        for i in valid_data_all:
            if i is not None:
                self.valid_data.append(i)
            else:
                self.valid_data.append(torch.zeros(self.nvalid))
        
    def get_train_batches(self, batch_size=None):
        
        # Set batch custom size
        if batch_size is None:
            batch_size = self.batch_size
        # Data needs to be translated to a tensor dataset that can then be
        # passed to the data loader
        
        tensor_data = data.TensorDataset(*self.train_data)
        # Shuffle training data and divide in batches
        train_batches = data.DataLoader(tensor_data, batch_size=batch_size,
                                        shuffle=True)
        
        # Get number of batches
        N_train_batches = int(np.ceil(self.ntrain/batch_size))
        
        return train_batches, N_train_batches
    
    def get_valid_batches(self,batch_size=None):
        
        # Set batch custom size
        if batch_size is None:
            batch_size = self.valid_batch_size
        #Transform data to tensor dataset
        tensor_data_v = data.TensorDataset(*self.valid_data)
        
        # Divide validation data into batches
        valid_batches = data.DataLoader(tensor_data_v,batch_size=batch_size)
        
        return valid_batches




    def _compute_E_statistics(self):
        x = self.E(self.idx_train)/self.N(self.idx_train).type(self.dtype)

        self._EperA_mean = (torch.sum(x, axis=0)/self.ntrain)
        self._EperA_stdev = (torch.sum((x - self.EperA_mean)**2, axis=0))
        self._EperA_stdev = (torch.sqrt(self.EperA_stdev/self.ntrain))
        return
    
    def _compute_F_statistics(self):
        self._FperA_mean  = 0.0
        self._FperA_stdev = 0.0
        for i in range(self.ntrain):
            F = self.F(i)
            x = 0.0
            for j in range(self.N(i)): 
                x = x + torch.sqrt(F[j][0]**2 + F[j][1]**2 + F[j][2]**2)
            m_prev = self.FperA_mean
            x = x/self.N(i).type(self.dtype)
            self._FperA_mean = (
                self.FperA_mean + (x - self.FperA_mean)/(i + 1))
            self._FperA_stdev = (
                self.FperA_stdev + (x - self.FperA_mean)*(x - m_prev))
        self._FperA_stdev = (torch.sqrt(self.FperA_stdev/self.ntrain))
        return
    
    def _compute_D_statistics(self):
        self._DperA_mean  = 0.0
        self._DperA_stdev = 0.0
        for i in range(self.ntrain):
            D = self.D(i)
            x = torch.sqrt(D[0]**2 + D[1]**2 + D[2]**2)
            m_prev = self.DperA_mean
            self._DperA_mean = (
                self.DperA_mean + (x - self.DperA_mean)/(i + 1))
            self._DperA_stdev = (
                self.DperA_stdev + (x - self.DperA_mean)*(x - m_prev))
        self._DperA_stdev = (torch.sqrt(self.DperA_stdev/self.ntrain))
        return
    
    @property 
    def EperA_mean(self):
        ''' Mean energy per atom in the training set '''
        if self._EperA_mean is None:
            self._compute_E_statistics()
        return self._EperA_mean

    @property
    def EperA_stdev(self): 
        ''' stdev of energy per atom in the training set '''
        if self._EperA_stdev is None:
            self._compute_E_statistics()
        return self._EperA_stdev
    
    @property 
    def FperA_mean(self): 
        ''' Mean force magnitude per atom in the training set '''
        if self._FperA_mean is None:
            self._compute_F_statistics()
        return self._FperA_mean

    @property
    def FperA_stdev(self): 
        ''' stdev of force magnitude per atom in the training set '''
        if self._FperA_stdev is None:
            self._compute_F_statistics()
        return self._FperA_stdev
    
    @property
    def DperA_mean(self): 
        ''' Mean partial charge per atom in the training set '''
        if self._DperA_mean is None:
            self._compute_D_statistics()
        return self._DperA_mean

    @property
    def DperA_stdev(self): 
        ''' stdev of partial charge per atom in the training set '''
        if self._DperA_stdev is None:
            self._compute_D_statistics()
        return self._DperA_stdev

    @property
    def EperA_m_n(self):
        if self._EperA_mean is None:
            self._compute_E_statistics()
        return np.float(self._EperA_mean.detach().numpy())

    @property
    def EperA_s_n(self):
        if self._EperA_stdev is None:
            self._compute_E_statistics()
        return np.float(self._EperA_stdev.detach().numpy())
    # @property
    # def train_data(self): 
    #     return self._DperA_stdev
        
