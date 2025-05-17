# python modules
import numpy as np
from numpy import exp, eye, zeros, ones, sqrt, mean, hstack, vstack
from scipy.sparse import eye as speye, csr_matrix

# package modules
from pandas import DataFrame

# local modules
from .SMap import SMap

#-----------------------------------------------------------
class HMSmap(SMap):
    '''HMSmap class : child of SMap
       
       This class extends SMap by incorporating a hidden Markov model 
       framework to filter observation noise using an Expectation 
       Maximization (EM) algorithm.
       
       When observation noise (vobs) is set to 0, HMSmap behaves exactly
       like standard SMap.
       
       HMSmap addresses a limitation in EDM by accounting for observation 
       noise, improving forecast accuracy and enabling robust computation 
       of Lyapunov exponents in noisy data.
    '''

    def __init__(self,
                 dataFrame       = None,
                 columns         = "",
                 target          = "",
                 lib             = "",
                 pred            = "",
                 E               = 0,
                 Tp              = 1,
                 knn             = 0,
                 tau             = -1,
                 theta           = 0.,
                 vobs            = 0.,           # Observation noise variance (new parameter)
                 kernel          = "gaussian",   # Kernel type (matches MATLAB implementation)
                 exclusionRadius = 0,
                 solver          = None,
                 embedded        = False,
                 validLib        = [],
                 noTime          = False,
                 generateSteps   = 0,
                 generateConcat  = False,
                 ignoreNan       = True,
                 maxIterations   = 100,          # EM algorithm max iterations
                 tolerance       = 0.01,         # EM convergence tolerance
                 inits           = None,         # Initial state estimates
                 figs            = 0,            # Plot flag
                 verbose         = False):
        '''Initialize HMSmap as child of SMap'''

        # Instantiate SMap parent class with standard parameters
        super(HMSmap, self).__init__(
            dataFrame       = dataFrame,
            columns         = columns,
            target          = target,
            lib             = lib,
            pred            = pred,
            E               = E,
            Tp              = Tp,
            knn             = knn,
            tau             = tau,
            theta           = theta,
            exclusionRadius = exclusionRadius,
            solver          = solver,
            embedded        = embedded,
            validLib        = validLib,
            noTime          = noTime,
            generateSteps   = generateSteps,
            generateConcat  = generateConcat,
            ignoreNan       = ignoreNan,
            verbose         = verbose
        )
        
        # Change name from 'SMap' to 'HMSmap'
        self.name = 'HMSmap'
        
        # HMSmap-specific parameters
        self.vobs          = vobs
        self.kernel        = kernel
        self.maxIterations = maxIterations
        self.tolerance     = tolerance
        self.inits         = inits
        self.figs          = figs
        
        # HMSmap-specific storage
        self.states        = None  # Filtered state estimates
        self.processVar    = None  # Process noise estimate
        self.stateHistory  = None  # History of state iterations

    #-------------------------------------------------------------------
    # Methods
    #-------------------------------------------------------------------
    def Run(self):
    #-------------------------------------------------------------------
        '''Run the HMSmap algorithm
        
        If vobs = 0, this just runs standard SMap.
        Otherwise, runs the EM algorithm to filter observation noise.
        '''
        # Setup steps from parent
        self.EmbedData()
        self.RemoveNan()
        
        # If no observation noise, just run standard SMap
        if self.vobs == 0:
            if self.verbose:
                print(f"{self.name}: No observation noise specified (vobs=0).")
                print(f"{self.name}: Running standard SMap algorithm.")
            
            # Run standard SMap algorithm
            self.FindNeighbors()
            self.Project()
            self.FormatProjection()
            
            # Set states equal to observations (no filtering)
            if hasattr(self, 'Projection') and self.Projection is not None:
                self.Projection['Filtered_States'] = self.Projection['Observations']
                # Move columns to match expected order
                cols = ['Time', 'Observations', 'Predictions', 'Filtered_States', 'Pred_Variance']
                self.Projection = self.Projection[cols]
            
            return
        
        # Otherwise, run EM algorithm for noise filtering
        self.HiddenMarkovSMap()
        self.FormatHMSOutput()
        
        # If plotting requested
        if self.figs == 1:
            self.PlotResults()

    #-------------------------------------------------------------------
    def SelectKernel(self):
    #-------------------------------------------------------------------
        """Select kernel function based on the specified type"""
        
        if self.verbose:
            print(f'{self.name}: SelectKernel()')
            
        # Define kernel function based on the specified type
        if self.kernel == 'exponential':
            return lambda D: exp(-self.theta * D)
        elif self.kernel == 'gaussian':
            return lambda D: exp(-self.theta**2 * (D)**2)
        elif self.kernel == 'inversedistance':
            return lambda D: 1.0 / (1 + self.theta**2 * (D)**2)
        elif self.kernel == 'quartic':
            return lambda D: exp(-self.theta**4 * (D)**4)
        else:
            # Default to gaussian
            if self.verbose:
                print(f"Warning: Unknown kernel '{self.kernel}', defaulting to gaussian")
            return lambda D: exp(-self.theta**2 * (D)**2)

    #-------------------------------------------------------------------
    def HiddenMarkovSMap(self):
    #-------------------------------------------------------------------
        """
        Implements the EM algorithm for the HMSmap model.
        This estimates states for an S-map model with observation and process uncertainty.
        """
        
        if self.verbose:
            print(f'{self.name}: HiddenMarkovSMap() - EM Algorithm')
        
        # Get the kernel function
        K = self.SelectKernel()
        
        # Initialize variables
        T = len(self.targetVec)
        
        # Initialize state estimates
        if self.inits is not None:
            xp = self.inits.copy()
        else:
            xp = self.targetVec.copy()
        
        # For scaling distances
        maxD = sqrt(self.E) * (np.max(self.targetVec) - np.min(self.targetVec))
        
        # Number of library points to use
        n = T - self.E * self.tau
        
        # Initialize tracking variables
        iter_count = 0
        delta = 10
        
        # Create identity matrices for calculation - EXACT MATCH TO MATLAB
        It = speye(T)                   # Sparse identity matrix of size T
        In = eye(n)                     # Dense identity matrix of size n
        
        # Initialize matrices - EXACT MATCH TO MATLAB
        # Build B as a dense matrix first, then convert to sparse
        B_top_dense = hstack((eye(self.E), zeros((self.E, n))))
        B_dense = vstack((B_top_dense, zeros((n, T))))
        B = csr_matrix(B_dense)
        
        A = zeros((T, 1))
        bj = zeros((self.E + 1, n))    # Matrix of regression coefficients
        
        # Track states over iterations
        Xtrack = []
        
        # Column of ones
        o_n = ones((n, 1))
        
        # Get data from targetVec
        x = self.targetVec.flatten()  # Convert to 1D array
        
        # Loop until convergence or max iterations
        while delta > self.tolerance and iter_count < self.maxIterations:
            iter_count += 1
            
            # Create lag matrix of embedded data
            xx = np.zeros((n, self.E + 1))
            
            # Fill the lag matrix with state estimates
            for j_idx in range(self.E + 1):
                start_idx = j_idx * self.tau
                xx[:, j_idx] = xp[start_idx:start_idx + n, 0]
                
            # H matrix has lagged values (and constant term)
            H = np.hstack((xx[:, 1:], ones((n, 1))))  # Add column of ones for intercept
            xj = xx[:, 0]  # First column (target variable)
            
            # M-step: optimize coefficients given state estimates
            vp = 0  # Process noise estimate
            
            for j_loop_idx in range(n):
                # Calculate distances and weights
                dj = np.sqrt(np.sum((H[j_loop_idx, :] - H)**2, axis=1)) / maxD
                min_nonzero = np.min(dj[dj > 0])
                wj = K(dj - min_nonzero)
                wj[j_loop_idx] = 0  # Zero weight for the point itself
                wj = wj / np.sum(wj)  # Normalize weights
                
                # Weighted SVD regression
                weighted_H = wj[:, np.newaxis] * H
                weighted_xj = wj * xj
                
                U, S, Vt = np.linalg.svd(weighted_H, full_matrices=False)
                # Handle small singular values
                S_inv = np.zeros_like(S)
                for k_idx in range(len(S)):
                    S_inv[k_idx] = 1.0 / S[k_idx] if S[k_idx] > 1e-5 else 0.0
                
                # Calculate coefficients
                bj[:, j_loop_idx] = Vt.T @ (S_inv[:, np.newaxis] * U.T) @ weighted_xj
                
                # Check for invalid coefficients
                if np.any(np.isinf(bj[:, j_loop_idx])) or np.any(np.isnan(bj[:, j_loop_idx])):
                    bj[:, j_loop_idx] = np.array([xj[j_loop_idx], 1.0] + [0.0] * (self.E - 1))
                
                # Update process noise estimate
                vp += (xj[j_loop_idx] - H[j_loop_idx, :] @ bj[:, j_loop_idx])**2 / n
                
                # Update B matrix for projection - MATCH MATLAB'S INDEXING
                for e_idx in range(self.E):
                    B[self.E + j_loop_idx, j_loop_idx + e_idx] = bj[self.E - e_idx, j_loop_idx]  # Reverse order as in MATLAB
            
            # Calculate variance ratio
            vrat = self.vobs / vp
            
            # E-step: compute expected value of x given b and observations
            xold = xp.copy()
            
            # Set intercept terms
            A[self.E:T, 0] = bj[0, :]
            
            # Calculate projection matrix
            Q = It - B
            
            if np.any(np.isinf(Q.data)) or np.any(np.isnan(Q.data)):
                print("Warning: Invalid values in projection matrix")
                break
                
            Ax = vrat * Q.T @ A
            
            # Update state estimates
            # (It + vrat * Q.T @ Q) is sparse, solve sparse linear system
            xp = np.linalg.solve((It + vrat * Q.T @ Q).toarray(), 
                                 (x.reshape(-1, 1) + Ax))
            
            # Maintain center and scale
            xp = xp * mean(xold) / mean(xp)
            
            # Save state history
            Xtrack.append(xp.copy().flatten())
            
            # Check convergence
            delta = np.max(np.abs(xp - xold))
            
            if self.verbose:
                print(f"Iteration {iter_count}, delta = {delta}, vp = {vp}")
        
        # Store state history
        self.stateHistory = np.column_stack(Xtrack) if Xtrack else np.array([])
        
        # After convergence, make predictions using S-map with filtered states
        # Store filtered states
        self.states = xp
        self.processVar = vp
        
        # Make one-step predictions using SMap approach but with filtered states
        self.MakePredictions(xp, bj, H, K, maxD)

    #-------------------------------------------------------------------
    def MakePredictions(self, xp, bj, H, K, maxD):
    #-------------------------------------------------------------------
        """
        Make one-step ahead predictions using the SMap approach
        with filtered states
        """
        T = len(self.targetVec)
        n = T - self.E * self.tau
        
        # Initialize predictions
        xpred = np.zeros((T, 1))
        xpred[:self.E] = xp[:self.E]  # Copy first E values
        
        # Make predictions for the rest
        for j_loop_idx in range(n):
            idx = self.E + j_loop_idx
            pred_idx = idx + self.Tp
            
            if pred_idx < T:
                # Get lagged values from filtered states
                lags = np.zeros(self.E)
                for e_idx in range(self.E):
                    lags[e_idx] = xp[idx - (e_idx + 1) * self.tau, 0]
                
                # Add intercept and make prediction
                pred_input = np.hstack((lags, 1.0))
                xpred[pred_idx, 0] = pred_input @ bj[:, j_loop_idx]
        
        # Store predictions
        self.projection = xpred.flatten()
        
        # Calculate errors
        self.pred_error = np.mean((self.targetVec.flatten()[self.E + self.Tp:T] - 
                                   xpred[self.E + self.Tp:T, 0])**2)
        self.obs_error = np.mean((self.targetVec.flatten() - xp.flatten())**2)
        
        # Multi-step ahead predictions if requested
        if self.generateSteps > 1:
            self.MultiStepPredict(self.targetVec.flatten(), xp.flatten(), H, K, maxD)

    #-------------------------------------------------------------------
    def MultiStepPredict(self, x, xp, H, K, maxD):
    #-------------------------------------------------------------------
        """
        Perform multi-step ahead predictions
        
        Args:
            x: Original time series
            xp: Filtered state estimates
            H: Matrix of lagged values
            K: Kernel function
            maxD: Scaling factor for distances
        """
        
        T = len(x)
        n = T - self.E * self.tau
        
        # Get the first column of lagged data
        xj = np.zeros(n)
        for i_idx in range(n):
            xj[i_idx] = xp[i_idx]
            
        # Initialize multistep predictions
        multistep_preds = np.zeros((T, self.generateSteps))
        multistep_errors = np.zeros(self.generateSteps)
        
        # First step predictions already computed in main algorithm
        multistep_preds[:, 0] = self.projection
        
        # Matrix for updating predictions
        Hahead = H.copy()
        
        # Calculate predictions for steps 2...stepsahead
        for k_step in range(1, self.generateSteps):
            # Update prediction vectors
            Hahead = np.hstack((multistep_preds[self.E:T, k_step-1].reshape(-1, 1), 
                              Hahead[:, :-2], 
                              np.ones((n, 1))))
            
            # Make predictions for each point
            for j_loop_idx in range(n):
                # Calculate distances and weights
                dj = np.sqrt(np.sum((Hahead[j_loop_idx, :] - H)**2, axis=1)) / maxD
                min_nonzero = np.min(dj[dj > 0])
                wj = K(dj - min_nonzero)
                wj = wj / np.sum(wj)
                
                # Weighted regression
                weighted_H = wj[:, np.newaxis] * H
                weighted_xj = wj * xj
                
                U, S, Vt = np.linalg.svd(weighted_H, full_matrices=False)
                
                # Handle small singular values
                S_inv = np.zeros_like(S)
                for i_svd in range(len(S)):
                    S_inv[i_svd] = 1.0 / S[i_svd] if S[i_svd] > 1e-5 else 0.0
                
                # Calculate multistep coefficients
                bja = Vt.T @ (S_inv[:, np.newaxis] * U.T) @ weighted_xj
                
                # Make k-step ahead prediction
                if self.E + j_loop_idx + k_step < T:
                    multistep_preds[self.E + j_loop_idx + k_step, k_step] = Hahead[j_loop_idx, :] @ bja
            
            # Calculate prediction errors for this step
            valid_indices = ~np.isnan(multistep_preds[self.E + k_step:T, k_step])
            if np.any(valid_indices):
                multistep_errors[k_step] = np.mean(
                    (x[self.E + k_step:T][valid_indices] - 
                     multistep_preds[self.E + k_step:T, k_step][valid_indices])**2)
        
        # Store multistep predictions
        self.multistep_predictions = multistep_preds
        self.multistep_errors = multistep_errors

    #-------------------------------------------------------------------
    def FormatHMSOutput(self):
    #-------------------------------------------------------------------
        """Format the HMS-map results for output"""
        
        if self.verbose:
            print(f'{self.name}: FormatHMSOutput()')
        
        # Create DataFrame for projections
        projection_df = DataFrame({
            'Time': self.time,
            'Observations': self.targetVec.flatten(),
            'Predictions': self.projection,
            'Filtered_States': self.states.flatten(),
            'Pred_Variance': np.full(len(self.time), np.nan)  # Placeholder for variance
        })
        
        # Create DataFrame for coefficients (similar to SMap)
        if self.tau < 0:
            coef_names = [f'∂{self.target[0]}/∂{self.target[0]}(t-{e})' for e in range(self.E)]
        else:
            coef_names = [f'∂{self.target[0]}/∂{self.target[0]}(t+{e})' for e in range(self.E)]
            
        # Store formatted results
        self.Projection = projection_df
        
        # Store additional metadata
        self.HMS_Info = {
            'Process_Variance': self.processVar,
            'Observation_Variance': self.vobs,
            'Prediction_Error': self.pred_error,
            'Observation_Error': self.obs_error
        }

    #-------------------------------------------------------------------
    def PlotResults(self):
    #-------------------------------------------------------------------
        """Plot the results of the HMSmap algorithm"""
        try:
            import matplotlib.pyplot as plt
            
            # Create figure for results
            plt.figure(figsize=(12, 8))
            
            # Plot multistep predictions if available
            if hasattr(self, 'multistep_predictions') and self.generateSteps > 1:
                for k_step in range(self.generateSteps):
                    # Original vs. predicted
                    plt.subplot(2, self.generateSteps, k_step + 1)
                    plt.plot(self.targetVec[self.E:-k_step-1], 
                             self.targetVec[self.E+k_step+1:], 'r.', 
                             label='Original')
                    plt.plot(self.states[self.E:-k_step-1], 
                             self.states[self.E+k_step+1:], 'b.', 
                             label='Filtered')
                    plt.title(f'{k_step+1}-step relation')
                    
                    if k_step == 0:
                        plt.legend()
                    
                    # Prediction performance
                    plt.subplot(2, self.generateSteps, self.generateSteps + k_step + 1)
                    actual_vals = self.targetVec[self.E+k_step+1:]
                    pred_vals = self.multistep_predictions[self.E+k_step+1:, k_step]
                    
                    valid = ~np.isnan(pred_vals)
                    if np.any(valid):
                        plt.plot(actual_vals[valid], 
                                 actual_vals[valid], 'k.')
                        plt.plot(actual_vals[valid], 
                                 pred_vals[valid], 'g.')
                        plt.title(f'{k_step+1}-step prediction')
            else:
                # Simple plot of observations vs filtered states
                plt.plot(self.time, self.targetVec, 'r.', label='Observations')
                plt.plot(self.time, self.states, 'b-', label='Filtered States')
                plt.plot(self.time, self.projection, 'g--', label='Predictions')
                plt.legend()
                plt.title('HMSmap Results')
                
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for plotting. Install with 'pip install matplotlib'")
            pass 