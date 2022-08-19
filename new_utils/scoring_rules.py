import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_cls_scores(input_matches, valid_idxs):
    """
    Computes proper scoring rule for multilabel classification results provided by retinanet.

    Args:
        input_matches (dict): dictionary containing input matches
        valid_idxs (tensor): a tensor containing valid element idxs for per-class computation
    Returns:
        output_dict (dict): dictionary containing ignorance and brier score.
    """

    output_dict = {}
    num_forecasts = input_matches['predicted_cls_probs'][valid_idxs].shape[0]

    # Construct binary probability vectors. Essential for RetinaNet as it uses
    # multilabel and not multiclass formulation.
    #escolher apenas os pares que pertencem a esta classe que esta a ser avaliada
    predicted_class_probs = input_matches['predicted_score_of_gt_category'][valid_idxs]

    # If no valid idxs, do not perform computation
    if predicted_class_probs.shape[0] == 0:
        output_dict.update({'ignorance_score_mean': None})
        return output_dict
    #Criar uma lista nx2, onde na primeira coluna temos o CS desta classe, e na 2ª coluna o inverso (1-CS).n é o numero de valid idxs
    predicted_multilabel_probs = torch.stack(
        [predicted_class_probs, 1.0 - predicted_class_probs], dim=1)
    #Criar uma lista nx2 onde na 1º colunas é so "1"(o valor que devia de ter sido obtido como CS) e na 2ª coluna "0"
    correct_multilabel_probs = torch.stack(
        [torch.ones(num_forecasts),
         torch.zeros(num_forecasts)], dim=1).to(device)

    #Calcular a primeira parte da Negative Log Likelihood (falta apenas a média de todos os pares).
    #Faz-se o logaritmo da probabilidade(CS) obtida, multiplica-se pela label correta (1 ou 0). Obtem se uma matrix nx2
    #na 1ªcoluna sao os valores relevantes, na 2ª coluna esta toda a zero porque multiplicou toda por zero.
    #O .sum(1) vai assim so remover a 2ª coluna
    predicted_log_likelihood_of_correct_category = (
        -correct_multilabel_probs * torch.log(predicted_multilabel_probs)).sum(1)
    #Agora sim, fazemos a media de todos os valores obtidos para cada previsao.
    #Atençao, este valor é efetivamente o NLL ou Ignorance Score, mas apenas para uma das classes!
    #So a seguir é que se tem de calcular a media para todas as classes
    cls_ignorance_score_mean = predicted_log_likelihood_of_correct_category.mean()
    output_dict.update({'ignorance_score_mean': cls_ignorance_score_mean.to(device).tolist()})

    return output_dict

def compute_cls_scores_fp(input_matches, valid_idxs):
    """
    Computes proper scoring rule for multilabel classification results provided by retinanet.

    Args:
        input_matches (dict): dictionary containing input matches
        valid_idxs (tensor): a tensor containing valid element idxs for per-class computation
    Returns:
        output_dict (dict): dictionary containing ignorance and brier score.
    """

    output_dict = {}
    num_forecasts = input_matches['predicted_cls_probs'][valid_idxs].shape[0]

    # Construct binary probability vectors. Essential for RetinaNet as it uses
    # multilabel and not multiclass formulation.
    #escolher apenas os pares que pertencem a esta classe que esta a ser avaliada
    predicted_class_probs = input_matches['predicted_score_of_gt_category'][valid_idxs]

    # If no valid idxs, do not perform computation
    if predicted_class_probs.shape[0] == 0:
        output_dict.update({'ignorance_score_mean': None})
        return output_dict
    #Criar uma lista nx2, onde na primeira coluna temos o CS desta classe, e na 2ª coluna o inverso (1-CS).n é o numero de valid idxs
    predicted_multilabel_probs = torch.stack(
        [predicted_class_probs, 1.0 - predicted_class_probs], dim=1)
    #Criar uma lista nx2 onde na 1º colunas é so "1"(o valor que devia de ter sido obtido como CS) e na 2ª coluna "0"
    correct_multilabel_probs = torch.stack(
        [torch.zeros(num_forecasts),
         torch.ones(num_forecasts)], dim=1).to(device)

    #Calcular a primeira parte da Negative Log Likelihood (falta apenas a média de todos os pares).
    #Faz-se o logaritmo da probabilidade(CS) obtida, multiplica-se pela label correta (1 ou 0). Obtem se uma matrix nx2
    #na 1ªcoluna sao os valores relevantes, na 2ª coluna esta toda a zero porque multiplicou toda por zero.
    #O .sum(1) vai assim so remover a 2ª coluna
    predicted_log_likelihood_of_correct_category = (
        -correct_multilabel_probs * torch.log(predicted_multilabel_probs)).sum(1)
    #Agora sim, fazemos a media de todos os valores obtidos para cada previsao.
    #Atençao, este valor é efetivamente o NLL ou Ignorance Score, mas apenas para uma das classes!
    #So a seguir é que se tem de calcular a media para todas as classes
    cls_ignorance_score_mean = predicted_log_likelihood_of_correct_category.mean()
    output_dict.update({'ignorance_score_mean': cls_ignorance_score_mean.to(device).tolist()})

    return output_dict

def compute_reg_scores(input_matches, valid_idxs):
    """
    Computes proper scoring rule for regression results.

    Args:
        input_matches (dict): dictionary containing input matches
        valid_idxs (tensor): a tensor containing valid element idxs for per-class computation

    Returns:
        output_dict (dict): dictionary containing ignorance and energy scores.
    """
    output_dict = {}
    #Recolher as informaçoes relevantes para esta classe
    #media das coordenadas obtidas assim como matriz de covariancias. E os valores das coordenadas das gts
    predicted_box_means = input_matches['predicted_box_means'][valid_idxs]
    predicted_box_covars = input_matches['predicted_box_covariances'][valid_idxs]
    gt_box_means = input_matches['gt_box_means'][valid_idxs]

    # If no valid idxs, do not perform computation
    if predicted_box_means.shape[0] == 0:
        output_dict.update({'ignorance_score_mean': None,
                            'mean_squared_error': None})
        return output_dict
    #Para cada prediction, criar umma distribuição multivariada normal (multivariada pois temos 4 variaveis),assim como temos matriz de covariancia tambem para essas variaveis
    #predicted_multivariate_normal_dists = torch.distributions.multivariate_normal.MultivariateNormal(
    #    predicted_box_means, predicted_box_covars + 1e-2 * torch.eye(predicted_box_covars.shape[2]).to(device))
    #Code(3 lines) that guarantee matrices to be positive definite
    diag_up = torch.triu(predicted_box_covars)
    upper = torch.triu(predicted_box_covars,diagonal=1)
    predicted_box_covars = torch.transpose(diag_up,1,2) + upper

    predicted_multivariate_normal_dists = torch.distributions.multivariate_normal.MultivariateNormal(
        predicted_box_means, predicted_box_covars + 1e-2 * torch.eye(predicted_box_covars.shape[2]).to(device))
    #apanha o logaritmo da probabilidade de uma determinada sample(ground truth) pertencer à distribuiçao prevista pela prediction 
    #feita pelo modelo (multivariate gaussian distributions com a media das 4 coordenadas e matriz de covariancia).
    #explicaçao aqui: https://stackoverflow.com/questions/54635355/what-does-log-prob-do
    #        if self._validate_args:
    #        self._validate_sample(value)
    #    diff = value - self.loc
    #    M = _batch_mahalanobis(self._unbroadcasted_scale_tril, diff)
    #    half_log_det = self._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    #    return -0.5 * (self._event_shape[0] * math.log(2 * math.pi) + M) - half_log_det
    
    negative_log_prob = - \
        predicted_multivariate_normal_dists.log_prob(gt_box_means)
    #print(negative_log_prob.max())
    #obtivemos uma lista com valores para a likelihood e agora é fazer a media desses valores todos
    negative_log_prob_mean = negative_log_prob.mean()
    output_dict.update({'ignorance_score_mean': negative_log_prob_mean.to(
        device).tolist()})

    mean_squared_error = ((predicted_box_means - gt_box_means)**2).mean()

    output_dict.update({'mean_squared_error': mean_squared_error.to(device).tolist(
    )})

    return output_dict

def is_pos_def(A):
    #if np.array_equal(A, A.T):
    if  torch.allclose(A, A.T):
        try:
            np.linalg.cholesky(A.cpu().numpy().astype('float64'))
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False
def compute_reg_scores_fp(false_positives, valid_idxs):
    """
    Computes proper scoring rule for regression false positives.

    Args:
        false_positives (dict): dictionary containing false_positives
        valid_idxs (tensor): a tensor containing valid element idxs for per-class computation

    Returns:
        output_dict (dict): dictionary containing false positives ignorance and energy scores.
    """
    output_dict = {}

    predicted_box_means = false_positives['predicted_box_means'][valid_idxs]
    predicted_box_covars = false_positives['predicted_box_covariances'][valid_idxs]
    #Code(3 lines) that guarantee matrices to be positive definite
    diag_up = torch.triu(predicted_box_covars)
    upper = torch.triu(predicted_box_covars,diagonal=1)
    predicted_box_covars = torch.transpose(diag_up,1,2) + upper

    #count = 0
    #for i , cov_matrix in enumerate(predicted_box_covars):
    #    cov_matrix_np = cov_matrix.cpu().numpy()        
    #    if not is_pos_def(cov_matrix):
    #        print(i)
    #        count += 1
    #        print(count)
    #        #teste = np.linalg.eigvals(cov_matrix_np)
    #        #temp = np.all(teste > 0)
    #        #if np.array_equal(A, A.T):
    #        if  torch.allclose(cov_matrix, cov_matrix.T):
    #            try:
    #                np.linalg.cholesky(cov_matrix_np.astype('float64'))
    #            except np.linalg.LinAlgError:
    #                print('É SIMETRICA, MAS NAO DA O CHOLESKY')
    #        else:
    #            print('NAO É SIMETRICA')
    predicted_multivariate_normal_dists = torch.distributions.multivariate_normal.MultivariateNormal(
        predicted_box_means, predicted_box_covars + 1e-2 * torch.eye(predicted_box_covars.shape[2]).to(device))

    # If no valid idxs, do not perform computation
    if predicted_box_means.shape[0] == 0:
        output_dict.update({'total_entropy_mean': None})
        return output_dict
    #The information entropy or entropy of a random variable is the average amount information or “surprise” due to the range values it can take
    #http://gregorygundersen.com/blog/2020/09/01/gaussian-entropy/
    total_entropy = predicted_multivariate_normal_dists.entropy()
    total_entropy_mean = total_entropy.mean()

    output_dict.update({'total_entropy_mean': total_entropy_mean.to(
        device).tolist()})

    return output_dict