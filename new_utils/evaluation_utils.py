from collections import defaultdict
import os
from matplotlib.pyplot import box
import torch
import json
import numpy as np
import tqdm
from detectron2.detectron2.structures import Boxes, pairwise_iou

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
def get_per_frame_preprocessed_instances():
    preprocess_gt_instances()
    preprocess_prediction_intances()
    return preprocessed_gt_instances, preprocessed_predicted_instances
"""
#Function that reads json file with labels from bdd and preprocesses them
#To obtain relevant information for metrics calculation
#Basically: For each image,  ground truth bboxes variables and associated classes.
def get_preprocess_ground_truth_instances(path_to_dataset_labels):
    try:
        #Load previously processed ground truth instances
        preprocessed_gt_instances = torch.load(
            os.path.join(path_to_dataset_labels,
            "preprocessed_gt_instances.pth"), 
            map_location=device)
        return preprocessed_gt_instances
    except FileNotFoundError:
        #If file does not exist yet, preprocess gt instances and save it.

        #Load json file with labels
        gt_info = json.load(
            open(os.path.join(path_to_dataset_labels,"val_coco_format.json"),"r")
            )
        #Get annotations from json file (list with all the ground truth bounding boxes)
        #Each Bbox has 4 variables for its location, the class, and the image associated with   
        gt_instances = gt_info['annotations']

        gt_boxes, gt_cat_idxs = defaultdict(torch.Tensor), defaultdict(torch.Tensor)
        for gt_instance in gt_instances:
            box_inds = gt_instance['bbox']
            #He does this transformation from [x,y,w,h] to [x1,y1,x2,y2] to both gt and predictions
            #Must be relevant/easier for metrics calculations
            box_inds = np.array([box_inds[0],
                                 box_inds[1],
                                 box_inds[0] + box_inds[2],
                                 box_inds[1] + box_inds[3]])
            #Append the new bbox instance to the list
            gt_boxes[gt_instance['image_id']] = torch.cat((gt_boxes[gt_instance['image_id']].cuda() , torch.as_tensor([box_inds],dtype=torch.float32).to(device)))

            gt_cat_idxs[gt_instance['image_id']] = torch.cat((gt_cat_idxs[gt_instance['image_id']].cuda() , torch.as_tensor([[gt_instance['category_id']]],dtype=torch.float32).to(device)))

        preprocessed_gt_instances = dict({'gt_boxes': gt_boxes,
                                          'gt_cat_idxs': gt_cat_idxs})

        torch.save(preprocessed_gt_instances,
        os.path.join(path_to_dataset_labels,
            "preprocessed_gt_instances.pth"),
        )
        return preprocessed_gt_instances

def get_preprocess_pred_instances(path_to_predictions_file):
    try:
        #Load previously processed predicted instances
        preprocessed_pred_instances = torch.load(
            os.path.join(path_to_predictions_file,
            "preprocessed_pred_instances.pth"), 
            map_location=device)
        return preprocessed_pred_instances
    except FileNotFoundError:

        predicted_instances = json.load(
            open(os.path.join(path_to_predictions_file, 'coco_instances_results.json'), "r")
            ) 

        predicted_boxes, predicted_cls_probs, predicted_covar_mats = defaultdict(torch.Tensor), defaultdict(torch.Tensor), defaultdict(torch.Tensor)

        is_odd = False
        min_allowed_score = 0
        for predicted_instance in predicted_instances:
            # Remove predictions with undefined category_id. This is used when the training and inference datasets come from
            # different data such as COCO-->VOC or BDD-->Kitti. Only happens if not ODD dataset, else all detections will
            # be removed.
            #PENSO QUE ESTE PASSO É REDUNDANTE, POIS AO GUARDAR O JSON EU JA SEPARO AS CLASSES RELEVANTES DAS NAO RELEVANTES
            #Se a categoria for -1 skip test, else verifica se o score obtido é acima do minimo, se nao for skip test
            if not is_odd:
                skip_test = (predicted_instance['category_id'] == -1) or (np.array(predicted_instance['cls_prob']).max(0) < min_allowed_score)
            else:
                skip_test = np.array(
                    predicted_instance['cls_prob']).max(0) < min_allowed_score

            if skip_test:
                continue

            box_inds = predicted_instance['bbox']
            #from top left corner,w,h to top left corner bottom right corner
            box_inds = np.array([box_inds[0],
                                 box_inds[1],
                                 box_inds[0] + box_inds[2],
                                 box_inds[1] + box_inds[3]])

            predicted_boxes[predicted_instance['image_id']] = torch.cat((predicted_boxes[predicted_instance['image_id']].to(
                device), torch.as_tensor([box_inds], dtype=torch.float32).to(device)))

            predicted_cls_probs[predicted_instance['image_id']] = torch.cat((predicted_cls_probs[predicted_instance['image_id']].to(
                device), torch.as_tensor([predicted_instance['cls_prob']], dtype=torch.float32).to(device)))

            #Converts covariance matrices from top-left corner and width-height representation to top-left bottom-right corner representation
            box_covar = np.array(predicted_instance['bbox_covar'])
            transformation_mat = np.array([[1.0, 0  , 0  , 0  ],
                                           [0  , 1.0, 0  , 0  ],
                                           [1.0, 0  , 1.0, 0  ],
                                           [0  , 1.0, 0.0, 1.0]])
            cov_pred = np.matmul(
                np.matmul(
                    transformation_mat,
                    box_covar),
                transformation_mat.T).tolist()

            predicted_covar_mats[predicted_instance['image_id']] = torch.cat(
                (predicted_covar_mats[predicted_instance['image_id']].to(device), torch.as_tensor([cov_pred], dtype=torch.float32).to(device)))

        preprocessed_pred_instances = dict({'predicted_boxes': predicted_boxes,
                                            'predicted_cls_probs': predicted_cls_probs,
                                            'predicted_covar_mats': predicted_covar_mats})

        torch.save(preprocessed_pred_instances,
        os.path.join(path_to_predictions_file,
            "preprocessed_pred_instances.pth"))
        
        return preprocessed_pred_instances


def get_matched_results(path_to_results, preprocessed_gt_instances, preprocessed_pred_instances):
    try:
        matched_results = torch.load(
            os.path.join(
                path_to_results,
                "matched_results.pth"), map_location=device)

        return matched_results
    
    except FileNotFoundError:
        predicted_cls_probs = preprocessed_pred_instances['predicted_cls_probs']
        predicted_box_means = preprocessed_pred_instances['predicted_boxes']
        predicted_box_covariances = preprocessed_pred_instances['predicted_covar_mats']
        gt_box_means = preprocessed_gt_instances['gt_boxes']
        gt_cat_idxs = preprocessed_gt_instances['gt_cat_idxs']

        true_positives = dict({'predicted_box_means': torch.Tensor().to(device),
                               'predicted_box_covariances': torch.Tensor().to(device),
                               'predicted_cls_probs': torch.Tensor().to(device),
                               'gt_box_means': torch.Tensor().to(device),
                               'gt_cat_idxs': torch.Tensor().to(device),
                               'iou_with_ground_truth': torch.Tensor().to(device)})

        duplicates = dict({'predicted_box_means': torch.Tensor().to(device),
                           'predicted_box_covariances': torch.Tensor().to(device),
                           'predicted_cls_probs': torch.Tensor().to(device),
                           'gt_box_means': torch.Tensor().to(device),
                           'gt_cat_idxs': torch.Tensor().to(device),
                           'iou_with_ground_truth': torch.Tensor().to(device)})

        false_positives = dict({'predicted_box_means': torch.Tensor().to(device),
                                'predicted_box_covariances': torch.Tensor().to(device),
                                'predicted_cls_probs': torch.Tensor().to(device)})

        false_negatives = dict({'gt_box_means': torch.Tensor().to(device),
                                'gt_cat_idxs': torch.Tensor().to(device)})
        #If there is a gt bbox where the iou < ioumin with all the predicted boxes, it means this gt bbox isnt predicted and is a false negative
        #If there is a predicted bbox where the iou < ioumin with all the gt boxes, it means this predicted bbox is a false positive

        iou_min = 0.1
        #iou that defines the true positive. a higher iou than this means that there is a great match with the ground truth
        iou_correct = 0.7
        
        with tqdm.tqdm(total=len(predicted_box_means)) as pbar:
            for key in predicted_box_means.keys():
                pbar.update(1)

                # Check if there are gt available for this image, 
                # if not all detections go to false positives
                if key not in gt_box_means.keys():
                    false_positives['predicted_box_means'] = torch.cat(
                        (false_positives['predicted_box_means'], predicted_box_means[key]))
                    false_positives['predicted_cls_probs'] = torch.cat(
                        (false_positives['predicted_cls_probs'], predicted_cls_probs[key]))
                    false_positives['predicted_box_covariances'] = torch.cat(
                        (false_positives['predicted_box_covariances'], predicted_box_covariances[key]))
                    continue

                # Compute iou between gt boxes and all predicted boxes in frame
                frame_gt_boxes = Boxes(gt_box_means[key])
                frame_predicted_boxes = Boxes(predicted_box_means[key])

                match_iou = pairwise_iou(frame_gt_boxes, frame_predicted_boxes)
                # Get false negative ground truth, which are fully missed.
                # These can be found by looking for ground truth boxes that have an
                # iou < iou_min with any detection
                #Neste caso (all(1)) vamos pegar em cada ground truth [21] e vamos comparar com todas as previsoes que existem.
                #Se todas forem abaixo de 0.1 (iou) entao nao ha nenhuma previsao que possivelmente represente esta ground truth e por isso é um false negative!
                false_negative_idxs = (match_iou <= iou_min).all(1)
                false_negatives['gt_box_means'] = torch.cat(
                    (false_negatives['gt_box_means'],
                    gt_box_means[key][false_negative_idxs]))
                false_negatives['gt_cat_idxs'] = torch.cat(
                    (false_negatives['gt_cat_idxs'],
                    gt_cat_idxs[key][false_negative_idxs]))

                # False positives are detections that have an iou < match iou with
                # any ground truth object.
                #POSSIVEL ERRO!!! nao devia ser entre iou_min e iou_correct?
                #NAO! Esta correto, o numero dentro do all() determina a dimensao onde vai buscar [21,100]. O all é uma funçao que "Tests if all elements in input evaluate to True."
                #Neste caso(all(0)) pega em cada previsao e vai comparar com todas as ground truths que existem. Se todas forem com iou abaixo de 0.1 entáo é um falso positivo, porque nao ha nenhuma gt que esta bbox esteja a representar
                #Basta haver uma gt que esta previsao tenha um iou>0.1 que ja é possivel ser true positive, (depois depende do criterio que utilizas para o classificar como tal, pode ser apenas um "duplicate")
                false_positive_idxs = (match_iou <= iou_min).all(0)
                false_positives['predicted_box_means'] = torch.cat(
                    (false_positives['predicted_box_means'],
                    predicted_box_means[key][false_positive_idxs]))
                false_positives['predicted_cls_probs'] = torch.cat(
                    (false_positives['predicted_cls_probs'],
                    predicted_cls_probs[key][false_positive_idxs]))
                false_positives['predicted_box_covariances'] = torch.cat(
                    (false_positives['predicted_box_covariances'],
                    predicted_box_covariances[key][false_positive_idxs]))

                # True positives are any detections with match iou > iou correct. We need to separate these detections to
                # True positive and duplicate set. The true positive detection is the detection assigned the highest score
                # by the neural network.
                #torch.nonzero :
                #Returns a tensor containing the indices of all non-zero elements of input. Each row in the result contains the indices of a non-zero element in input.
                #If input has n dimensions, then the resulting indices tensor out is of size (z×n) 
                #where z is the total number of non-zero elements in the input tensor.
                #Neste caso imaginemos que temos a matriz 21x100. temos 2 dimensoes, o resultado vai ser zx2, onde z é o numero de indexes non zero
                #o z é igual porque estamos a encontrar o tal numero non zero na matriz de 2 dimensoes, um valor diz-te onde esta na linha, outro diz-te onde etsa na coluna
                true_positive_idxs = torch.nonzero(match_iou >= iou_correct)

                # Setup tensors to allow assignment of detections only once.
                gt_idxs_processed = torch.tensor(
                    []).type(torch.LongTensor).to(device)

                for i in torch.arange(frame_gt_boxes.tensor.shape[0]):
                    #Este loop vai iterar cada uma das gt bbox
                    #Neste ,momento temos os indexes dos true positives, onde os podemos encontrar na matriz. Queremos agora ir gt a gt
                    # encontrar o true positive dessa gt, fazer essa "conexão" e todas as outras previsoes que tambem sao true positive
                    #desta gt, mas possuem um confidence score menor, sao considerados "duplicates" e postos à parte.
                    # Check if true positive has been previously assigned to a ground truth box and remove it if this is
                    # the case. Very rare occurrence but need to handle it
                    # nevertheless.

                    #vai buscar os indexes das gt boxes onde ha non zeros.
                    #depois de os ter, vai confirmar se algum destes indexes corresponde à gt que estamos a "investigar de momento"
                    #Havendo algum, vamos entao filtrar a matriz dos true positives index para obter apenas a informaçao relativa a essa gt
                    #E por fim no gt_idxs obtemos entao os indexes das predictions que sao true positive e correspondem a esta gt.
                    gt_idxs = true_positive_idxs[true_positive_idxs[:, 0] == i][:, 1]
                    #Supostamente esta parte era para garantir que se uma prediction box ja tivesse sido atribuida a uma gt
                    #entao ia ser utilizada a seguir. No entanto ele parece nunca atualizar o "gt_idxs_processed"
                    #por isso acho que esta parte do codigo acaba por nao fazer nada.
                    non_valid_idxs = torch.nonzero(
                        gt_idxs_processed[..., None] == gt_idxs)

                    if non_valid_idxs.shape[0] > 0:
                        gt_idxs[non_valid_idxs[:, 1]] = -1
                        gt_idxs = gt_idxs[gt_idxs != -1]

                    #Se houverem predictions que sao true positive(pelo menos uma)
                    if gt_idxs.shape[0] > 0:
                        #Vamos pegar nessas predictions e analisar os resultados obtidos para as classes
                        current_matches_predicted_cls_probs = predicted_cls_probs[key][gt_idxs]
                        #obtem se os valores maximos de confidecence score para cada prediction, assim como o index da classe
                        max_score, _ = torch.max(
                            current_matches_predicted_cls_probs, 1)
                        #Ordena os confidence scores maximos de cada prediction e os seus indexes
                        _, max_idxs = max_score.topk(max_score.shape[0])

                        #se houver mais do que uma prediction possivel para true posive
                        if max_idxs.shape[0] > 1:
                            #o index com melhor score vai ser o verdadeiro true postive
                            max_idx = max_idxs[0]
                            #todos os outros serao considerados duplicates
                            duplicate_idxs = max_idxs[1:]
                        else:
                            #no caso de apenas haver um unico true positive, a lista dos duplicates fica vazia
                            max_idx = max_idxs
                            duplicate_idxs = torch.empty(0).to(device)

                        #vamos buscar aqui as informaçoes que faltavam dos true positives desta gt, coordenadas e matriz de cov
                        current_matches_predicted_box_means = predicted_box_means[key][gt_idxs]
                        current_matches_predicted_box_covariances = predicted_box_covariances[
                            key][gt_idxs]

                        # Highest scoring detection goes to true positives
                        true_positives['predicted_box_means'] = torch.cat(
                            (true_positives['predicted_box_means'],
                            current_matches_predicted_box_means[max_idx:max_idx + 1, :]))
                        true_positives['predicted_cls_probs'] = torch.cat(
                            (true_positives['predicted_cls_probs'],
                            current_matches_predicted_cls_probs[max_idx:max_idx + 1, :]))
                        true_positives['predicted_box_covariances'] = torch.cat(
                            (true_positives['predicted_box_covariances'],
                            current_matches_predicted_box_covariances[max_idx:max_idx + 1, :]))

                        true_positives['gt_box_means'] = torch.cat(
                            (true_positives['gt_box_means'], gt_box_means[key][i:i + 1, :]))
                        true_positives['gt_cat_idxs'] = torch.cat(
                            (true_positives['gt_cat_idxs'], gt_cat_idxs[key][i:i + 1, :]))
                        true_positives['iou_with_ground_truth'] = torch.cat(
                            (true_positives['iou_with_ground_truth'], match_iou[i, gt_idxs][max_idx:max_idx + 1]))

                        # Lower scoring redundant detections go to duplicates
                        if duplicate_idxs.shape[0] > 1:
                            duplicates['predicted_box_means'] = torch.cat(
                                (duplicates['predicted_box_means'], current_matches_predicted_box_means[duplicate_idxs, :]))
                            duplicates['predicted_cls_probs'] = torch.cat(
                                (duplicates['predicted_cls_probs'], current_matches_predicted_cls_probs[duplicate_idxs, :]))
                            duplicates['predicted_box_covariances'] = torch.cat(
                                (duplicates['predicted_box_covariances'],
                                current_matches_predicted_box_covariances[duplicate_idxs, :]))
                            #o np.repeat é utilizado para ficar com uma lista onde repetes varias vezes a gt consoante o numero de previsoes
                            #que foram associadas a esta gt
                            duplicates['gt_box_means'] = torch.cat(
                                (duplicates['gt_box_means'], gt_box_means[key][np.repeat(i, duplicate_idxs.shape[0]), :]))
                            duplicates['gt_cat_idxs'] = torch.cat(
                                (duplicates['gt_cat_idxs'], gt_cat_idxs[key][np.repeat(i, duplicate_idxs.shape[0]), :]))
                            duplicates['iou_with_ground_truth'] = torch.cat(
                                (duplicates['iou_with_ground_truth'],
                                match_iou[i, gt_idxs][duplicate_idxs]))

                        elif duplicate_idxs.shape[0] == 1:
                            # Special case when only one duplicate exists, required to
                            # index properly for torch.cat
                            duplicates['predicted_box_means'] = torch.cat(
                                (duplicates['predicted_box_means'],
                                current_matches_predicted_box_means[duplicate_idxs:duplicate_idxs + 1, :]))
                            duplicates['predicted_cls_probs'] = torch.cat(
                                (duplicates['predicted_cls_probs'],
                                current_matches_predicted_cls_probs[duplicate_idxs:duplicate_idxs + 1, :]))
                            duplicates['predicted_box_covariances'] = torch.cat(
                                (duplicates['predicted_box_covariances'],
                                current_matches_predicted_box_covariances[duplicate_idxs:duplicate_idxs + 1, :]))

                            duplicates['gt_box_means'] = torch.cat(
                                (duplicates['gt_box_means'], gt_box_means[key][i:i + 1, :]))
                            duplicates['gt_cat_idxs'] = torch.cat(
                                (duplicates['gt_cat_idxs'], gt_cat_idxs[key][i:i + 1, :]))
                            duplicates['iou_with_ground_truth'] = torch.cat(
                                (duplicates['iou_with_ground_truth'],
                                match_iou[i, gt_idxs][duplicate_idxs:duplicate_idxs + 1]))

        matched_results = dict()
        #no fim de tudo vamos ter um dicionario onde nos "true positives" temos uma razão de 1gt para 1 detection que possui mais de 0.7 de iou
        #duplicates: razao de 1 para 1 de gt para detections que sao redundantes e estao a classificar a mesma gt que ja foi detetatada no true positives
        #mas atençaoq ue todos estes duplicates sao tambem true positives para esta gt, simplesmente tem um score pior.
        #false positives: apenas predictions que nao estao a representar nenhuma gt existente
        #false negatives : apenas gt que nao houve qualquer deteção relevante
        matched_results.update({"true_positives": true_positives,
                                "duplicates": duplicates,
                                "false_positives": false_positives,
                                "false_negatives": false_negatives})
        torch.save(
            matched_results,
            os.path.join(
                path_to_results,
                "matched_results.pth"))

        return matched_results