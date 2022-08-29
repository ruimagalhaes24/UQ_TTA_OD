import torch
from prettytable import PrettyTable
from new_utils.evaluation_utils import is_pos_def
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def remove_detections(outputs):
    """
    This functions removes uncertain bbox predictions based on a threshold for the accepted uncertainty.
    Hopefully it will remove false positive detections, while mantaining true positives.
    See the following explanations: https://online.stat.psu.edu/stat505/book/export/html/645
    Total Variation(Variance) equals to the sum of all diagonal elements of the covariance matrix
    Generalized Variance is equal to the determinant of the covariance matrix. It will take into account the correlation between the variables
    Where as the Total Variance assumes independent variables. 

    Args:
    -List of outputs after Uncertainty Method(TTA, MC dropout, Output Redundancy)
    Returns:
    -Final list of outputs where uncertain ones were removed
    """
    generalized_variance = torch.tensor([])
    for prediction in outputs:
        teste = prediction['bbox_covar']
        teste = torch.tensor(prediction['bbox_covar'])
        teste = torch.unsqueeze(torch.det(torch.tensor(prediction['bbox_covar'])), dim=0)
        generalized_variance = torch.cat((generalized_variance, torch.unsqueeze(torch.det(torch.tensor(prediction['bbox_covar'])), dim=0)))
    return None

def obtain_uncertainty_statistics(matches):
    """
    Function to understand uncertaintities in True Positives, False Positives
    """
    
    #True Positives Média obtida para OP:48.7
    tp_cov_matrices = matches['true_positives']['predicted_box_covariances']
    #Total Variance TP
    tp_total_variance = torch.sum(torch.diagonal(tp_cov_matrices,dim1=1,dim2=2),dim=1)
    tp_total_variance_mean = torch.mean(tp_total_variance)
    tp_total_variance_median = torch.median(tp_total_variance)
    #Generalized Variance TP
    tp_generalized_variance = torch.det(tp_cov_matrices)
    tp_mean_generalized_variance = torch.mean(tp_generalized_variance)
    tp_median_generalized_variance = torch.median(tp_generalized_variance)
    
    #Code(3 lines) that guarantee matrices to be positive definite
    diag_up = torch.triu(tp_cov_matrices)
    upper = torch.triu(tp_cov_matrices,diagonal=1)
    tp_cov_matrices = torch.transpose(diag_up,1,2) + upper

    count = 0
    for i , cov_matrix in enumerate(tp_cov_matrices):
        cov_matrix_np = cov_matrix.cpu().numpy()        
        if not is_pos_def(cov_matrix):
            print(i)
            count += 1
            print(count)
            #teste = np.linalg.eigvals(cov_matrix_np)
            #temp = np.all(teste > 0)
            #if np.array_equal(A, A.T):
            if  torch.allclose(cov_matrix, cov_matrix.T):
                try:
                    np.linalg.cholesky(cov_matrix_np.astype('float64'))
                except np.linalg.LinAlgError:
                    print('É SIMETRICA, MAS NAO DA O CHOLESKY')
            else:
                print('NAO É SIMETRICA')  

    distributions_tp = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(
                tp_cov_matrices.shape[0:2]).to(device), tp_cov_matrices + 1e-4 * torch.eye(tp_cov_matrices.shape[2]).to(device))

    entropy_tp = distributions_tp.entropy()
    entropy_mean_tp = torch.mean(entropy_tp)
    entropy_median_tp = torch.median(entropy_tp)

    #Duplicates Média obtida para OP: 19.7
    dup_cov_matrices = matches['duplicates']['predicted_box_covariances']
    #Total Variance Dup
    dup_total_variance = torch.sum(torch.diagonal(dup_cov_matrices,dim1=1,dim2=2),dim=1)
    dup_total_variance_mean = torch.mean(dup_total_variance)
    dup_total_variance_median = torch.median(dup_total_variance)
    #Generalized Variance Dup
    dup_generalized_variance = torch.det(dup_cov_matrices)
    dup_mean_generalized_variance = torch.mean(dup_generalized_variance)
    dup_median_generalized_variance = torch.median(dup_generalized_variance)

    ##Code(3 lines) that guarantee matrices to be positive definite
    diag_up = torch.triu(dup_cov_matrices)
    upper = torch.triu(dup_cov_matrices,diagonal=1)
    dup_cov_matrices = torch.transpose(diag_up,1,2) + upper

    distributions_dup = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(
                dup_cov_matrices.shape[0:2]).to(device), dup_cov_matrices + 1e-4 * torch.eye(dup_cov_matrices.shape[2]).to(device))
    
    entropy_dup = distributions_dup.entropy()
    entropy_mean_dup = torch.mean(entropy_dup)
    entropy_median_dup = torch.median(entropy_dup)

    #False Positives Média obtida para OP: 2449.43
    fp_cov_matrices = matches['false_positives']['predicted_box_covariances']
    #Total Variance FP
    fp_total_variance = torch.sum(torch.diagonal(fp_cov_matrices,dim1=1,dim2=2),dim=1)
    fp_total_variance_mean = torch.mean(fp_total_variance)
    fp_total_variance_median = torch.median(fp_total_variance)
    #Generalized Variance FP
    fp_generalized_variance = torch.det(fp_cov_matrices)
    fp_mean_generalized_variance = torch.mean(fp_generalized_variance)
    fp_median_generalized_variance = torch.median(fp_generalized_variance)
    #Code(3 lines) that guarantee matrices to be positive definite
    diag_up = torch.triu(fp_cov_matrices)
    upper = torch.triu(fp_cov_matrices,diagonal=1)
    fp_cov_matrices = torch.transpose(diag_up,1,2) + upper
    
    distributions_fp = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(
                fp_cov_matrices.shape[0:2]).to(device), fp_cov_matrices + 1e-4 * torch.eye(fp_cov_matrices.shape[2]).to(device))
    
    entropy_fp = distributions_fp.entropy()
    entropy_mean_fp = torch.mean(entropy_fp)
    entropy_median_fp = torch.median(entropy_fp)

    tp_prob_vectors = matches['true_positives']['predicted_cls_probs']
    tp_shannon_entropy_dists = torch.distributions.categorical.Categorical(tp_prob_vectors)
    tp_shannon_entropy = tp_shannon_entropy_dists.entropy()
    tp_shannon_entropy_mean = torch.mean(tp_shannon_entropy)
    tp_shannon_entropy_median = torch.median(tp_shannon_entropy)

    dup_prob_vectors = matches['duplicates']['predicted_cls_probs']
    dup_shannon_entropy_dists = torch.distributions.categorical.Categorical(dup_prob_vectors)
    dup_shannon_entropy = dup_shannon_entropy_dists.entropy()
    dup_shannon_entropy_mean = torch.mean(dup_shannon_entropy)
    dup_shannon_entropy_median = torch.median(dup_shannon_entropy)

    fp_prob_vectors = matches['false_positives']['predicted_cls_probs']
    fp_shannon_entropy_dists = torch.distributions.categorical.Categorical(fp_prob_vectors)
    fp_shannon_entropy = fp_shannon_entropy_dists.entropy()
    fp_shannon_entropy_mean = torch.mean(fp_shannon_entropy)
    fp_shannon_entropy_median = torch.median(fp_shannon_entropy)

    print(torch.sum(fp_generalized_variance > 0.5))

    table = PrettyTable()
    table.field_names = (['TP Mean TV',
                          'TP Median TV',
                          'Dup Mean TV',
                          'Dup Median TV',
                          'FP Mean TV',
                          'FP Median TV'])
    table.add_row(['{:.4f}'.format(tp_total_variance_mean),
                   '{:.4f}'.format(tp_total_variance_median),
                   '{:.4f}'.format(dup_total_variance_mean),
                   '{:.4f}'.format(dup_total_variance_median),
                   '{:.4f}'.format(fp_total_variance_mean),
                   '{:.4f}'.format(fp_total_variance_median)])
    print(table)
    table = PrettyTable()
    table.field_names = (['TP Mean GV',
                          'TP Median GV',
                          'Dup Mean GV',
                          'Dup Median GV',
                          'FP Mean GV',
                          'FP Median GV'])
    table.add_row(['{:.4f}'.format(tp_mean_generalized_variance),
                   '{:.4f}'.format(tp_median_generalized_variance),
                   '{:.4f}'.format(dup_mean_generalized_variance),
                   '{:.4f}'.format(dup_median_generalized_variance),
                   '{:.4f}'.format(fp_mean_generalized_variance),
                   '{:.4f}'.format(fp_median_generalized_variance)])
    print(table)
    table = PrettyTable()
    table.field_names = (['TP Mean entropy',
                          'TP Median entropy',
                          'Dup Mean entropy',
                          'Dup Median entropy',
                          'FP Mean entropy',
                          'FP Median entropy'])
    table.add_row(['{:.4f}'.format(entropy_mean_tp),
                   '{:.4f}'.format(entropy_median_tp),
                   '{:.4f}'.format(entropy_mean_dup),
                   '{:.4f}'.format(entropy_median_dup),
                   '{:.4f}'.format(entropy_mean_fp),
                   '{:.4f}'.format(entropy_median_fp)])
    print(table)
    return None