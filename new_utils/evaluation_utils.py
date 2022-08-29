from collections import defaultdict
import os
from matplotlib.pyplot import box
import torch
import json
import numpy as np
import tqdm
from detectron2.detectron2.structures import Boxes, pairwise_iou

import new_utils.scoring_rules
#pip3 install uncertainty-calibration
import calibration as cal
from prettytable import PrettyTable

# Coco evaluator tools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import copy
from new_utils.scoring_rules import is_pos_def

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
            open(os.path.join(path_to_predictions_file, 'coco_instances_results_xyxy.json'), "r")
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
            #box_inds = np.array([box_inds[0],
            #                     box_inds[1],
            #                     box_inds[0] + box_inds[2],
            #                     box_inds[1] + box_inds[3]])
            box_inds = np.array(box_inds)

            predicted_boxes[predicted_instance['image_id']] = torch.cat((predicted_boxes[predicted_instance['image_id']].to(
                device), torch.as_tensor([box_inds], dtype=torch.float32).to(device)))

            predicted_cls_probs[predicted_instance['image_id']] = torch.cat((predicted_cls_probs[predicted_instance['image_id']].to(
                device), torch.as_tensor([predicted_instance['cls_prob']], dtype=torch.float32).to(device)))

            #Converts covariance matrices from top-left corner and width-height representation to top-left bottom-right corner representation
            box_covar = np.array(predicted_instance['bbox_covar'])
            #transformation_mat = np.array([[1.0, 0  , 0  , 0  ],
            #                               [0  , 1.0, 0  , 0  ],
            #                               [1.0, 0  , 1.0, 0  ],
            #                               [0  , 1.0, 0.0, 1.0]])
            #cov_pred = np.matmul(
            #    np.matmul(
            #        transformation_mat,
            #        box_covar),
            #    transformation_mat.T).tolist()
            cov_pred = box_covar
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

def compute_nll(matched_results_,kitti = False):
    #Category mapping YOLO to BDD
    #cat_mapping_dict = {2: 1, 5: 2, 7: 3, 0: 4, 1: 6, 3: 7}
    #É preciso ter em atençao que nas ground truths a categoria "rider" nao existe no yolo. sera que devia entao de tirá-la das gt???
    #Ja reparei em algumas imagens que ele classifica o rider como sendo "person". Outra opçao tambem poderia ser colocar a label como "motorcycle"
    #aqui vou ter que colocar o equivalente da label no vetor de 80 classes do yolo
    #YOLO | BDD | kitti
    #2 car | 1 car | 1 car
    #5 bus | 2 bus | ----
    #7 truck| 3 truck | 2 truck
    #0 person| 4 person | 3 person
    #-1 rider| 5 rider | 4 rider
    #1 bycicle | 6 bike | 5 bycicle
    #3 motorcycle | 7 motor | ---

    matched_results = copy.deepcopy(matched_results_)
    if kitti:
        cat_mapping_dict = {5: 1, 4: 0, 3: 0, 2: 7, 1: 2}
    else:
        cat_mapping_dict = {7: 3, 6: 1, 5: 0, 4: 0, 3: 7, 2: 5, 1: 2}
    #cat_mapping_dict = {1:1,2:2,3:3,4:4,5:5,6:6,7:7}
    with torch.no_grad():
        # Build preliminary dicts required for computing classification scores.
        for matched_results_key in matched_results.keys():
            #Nao percebo porque é que a parte do category mappiung é necessaria se as predictions ja vem correto.
        #Esta parte é necessaria porque pode ser necessario converter as ground truths para as categorias de um dataset especifico, tipo kitti ou lyft! 
            if 'gt_cat_idxs' in matched_results[matched_results_key].keys():
                # First we convert the written things indices to contiguous
                # indices.
                gt_converted_cat_idxs = matched_results[matched_results_key]['gt_cat_idxs'].squeeze(
                    1)
                #ele converte as classes de 1,2,3,4,5,6,7 para 01,2,3,4,5,6. Sera que isto é apenas e só porque o output final das classes é um vetor com 7 valores
                #e por isso quer fazer essa equivalencia? Sim é! Isto porque esse é o formato do output final do retinanet
                #NEste caso tu tens de fazer o equivalente mas para o YOLO, tens de ir buscar os indexes na lista das classes que contem os cs que queres
                gt_converted_cat_idxs = torch.as_tensor([cat_mapping_dict[class_idx.cpu(
                ).tolist()] for class_idx in gt_converted_cat_idxs]).to(device)
                matched_results[matched_results_key]['gt_converted_cat_idxs'] = gt_converted_cat_idxs.to(
                    device)
                #print(torch.unique(gt_converted_cat_idxs))
                if 'predicted_cls_probs' in matched_results[matched_results_key].keys(
                ):
                    predicted_cls_probs = matched_results[matched_results_key]['predicted_cls_probs']
                    # This is required for evaluation of retinanet based
                    # detections.
                    #Este passo aqui vai obter o CS que a previsao tem para a categoria que é suposto ser da ground truth
                    matched_results[matched_results_key]['predicted_score_of_gt_category'] = torch.gather(
                        predicted_cls_probs, 1, gt_converted_cat_idxs.unsqueeze(1)).squeeze(1)
                matched_results[matched_results_key]['gt_cat_idxs'] = gt_converted_cat_idxs
            else:
                # For false positives, the correct category is background. For retinanet, since no explicit
                # background category is available, this value is computed as 1.0 - score of the predicted
                # category.
                predicted_class_probs, predicted_class_idx = matched_results[matched_results_key]['predicted_cls_probs'].max(
                    1)
                #como previamente ja tinha feito uma seleçao das instances com as categorias relevantes, neste passo apenas vamos ter 
                #as categorias relevantes do yolo!! :)
                #print(torch.unique(predicted_class_idx))
                matched_results[matched_results_key]['predicted_score_of_gt_category'] = 1.0 - \
                    predicted_class_probs
                matched_results[matched_results_key]['predicted_cat_idxs'] = predicted_class_idx
        
        # Load the different detection partitions
        true_positives = matched_results['true_positives']
        false_negatives = matched_results['false_negatives']
        false_positives = matched_results['false_positives']

        # Get the number of elements in each partition
        num_true_positives = true_positives['predicted_box_means'].shape[0]
        num_false_negatives = false_negatives['gt_box_means'].shape[0]
        num_false_positives = false_positives['predicted_box_means'].shape[0]

        per_class_output_list = []
        if kitti:
            meta_catalog = [0,1,2,7]
        else:    
            meta_catalog = [0,1,2,3,5,7]
        #print(torch.unique(true_positives['gt_converted_cat_idxs']))
        #print(torch.unique(false_positives['predicted_cat_idxs']))
        for class_idx in meta_catalog:
            #Encontrar os TP e FP para esta classe
            true_positives_valid_idxs = true_positives['gt_converted_cat_idxs'] == class_idx
            false_positives_valid_idxs = false_positives['predicted_cat_idxs'] == class_idx

            # Compute classification metrics for every partition
            true_positives_cls_analysis = new_utils.scoring_rules.compute_cls_scores(
                true_positives, true_positives_valid_idxs)
            #alterei a funçao, pois o confidence score que os fp obtem, esta relacionado com o target 0(visto nao haver nada la)
            #e nao com o target 1
            false_positives_cls_analysis = new_utils.scoring_rules.compute_cls_scores_fp(
                false_positives, false_positives_valid_idxs)

            # Compute regression metrics for every partition
            true_positives_reg_analysis = new_utils.scoring_rules.compute_reg_scores(
                true_positives, true_positives_valid_idxs)
            false_positives_reg_analysis = new_utils.scoring_rules.compute_reg_scores_fp(
                false_positives, false_positives_valid_idxs)
            #false_positives_reg_analysis = {'ignorance_mean':0}

            per_class_output_list.append(
                {'true_positives_cls_analysis': true_positives_cls_analysis,
                    'true_positives_reg_analysis': true_positives_reg_analysis,
                    'false_positives_cls_analysis': false_positives_cls_analysis,
                    'false_positives_reg_analysis': false_positives_reg_analysis})
        
        final_accumulated_output_dict = dict()
        final_average_output_dict = dict()

        #visto que ate este ponto temos a analise feita por classe, agora falta fazer a media de todos os valores obtidos.
        for key in per_class_output_list[0].keys():
            average_output_dict = dict()
            for inner_key in per_class_output_list[0][key].keys():
                collected_values = [per_class_output[key][inner_key]
                                    for per_class_output in per_class_output_list if per_class_output[key][inner_key] is not None]
                collected_values = np.array(collected_values)

                if key in average_output_dict.keys():
                    # Use nan mean since some classes do not have duplicates for
                    # instance or has one duplicate for instance. torch.std returns nan in that case
                    # so we handle those here. This should not have any effect on the final results, as
                    # it only affects inter-class variance which we do not
                    # report anyways.
                    average_output_dict[key].update(
                        {inner_key: np.nanmean(collected_values)})
                    final_accumulated_output_dict[key].update(
                        {inner_key: collected_values})
                else:
                    average_output_dict.update(
                        {key: {inner_key: np.nanmean(collected_values)}})
                    final_accumulated_output_dict.update(
                        {key: {inner_key: collected_values}})

            final_average_output_dict.update(average_output_dict)
    table = PrettyTable()
    table.field_names = (['TP Cls NLL',
                          'TP Reg NLL',
                          'TP Reg MSE',
                          'FP Cls NLL',
                          'FP Reg Total Entropy Mean'])
    table.add_row(['{:.4f}'.format(final_average_output_dict['true_positives_cls_analysis']['ignorance_score_mean']),
                   '{:.4f}'.format(final_average_output_dict['true_positives_reg_analysis']['ignorance_score_mean']),
                   '{:.4f}'.format(final_average_output_dict['true_positives_reg_analysis']['mean_squared_error']),
                   '{:.4f}'.format(final_average_output_dict['false_positives_cls_analysis']['ignorance_score_mean']),
                   '{:.4f}'.format(final_average_output_dict['false_positives_reg_analysis']['total_entropy_mean'])])
    print(table)
    table = PrettyTable()
    table.field_names = (['Output Type',
                              'Number of Instances',
                              'Cls Ignorance Score',
                              'Reg Ignorance Score'])
    table.add_row(
            [
                "True Positives:",
                num_true_positives,
                '{:.4f}'.format(
                    final_average_output_dict['true_positives_cls_analysis']['ignorance_score_mean']),
                '{:.4f}'.format(
                    final_average_output_dict['true_positives_reg_analysis']['ignorance_score_mean'])])

    table.add_row(
            [
                "False Positives:",
                num_false_positives,
                '{:.4f}'.format(
                    final_average_output_dict['false_positives_cls_analysis']['ignorance_score_mean']),
                '{:.4f}'.format(
                    final_average_output_dict['false_positives_reg_analysis']['total_entropy_mean'])])

    table.add_row(["False Negatives:",
                       num_false_negatives,
                       '-',
                       '-'])
    print(table)
    return final_average_output_dict, final_accumulated_output_dict

def compute_calibration_uncertainty_errors(matched_results,kitti):
    #YOLO | BDD | kitti
    #2 car | 1 car | 1 car
    #5 bus | 2 bus | ----
    #7 truck| 3 truck | 2 truck
    #0 person| 4 person | 3 person
    #-1 rider| 5 rider | 4 rider
    #1 bycicle | 6 bike | 5 bycicle
    #3 motorcycle | 7 motor | ---

    #As labels da gt sao :                 45,6,1,7,2,3
    #As previsoes do yolo estao nos lugares 0,1,2,3,5,7
    #Ao escolher apenas estas vamos para:   0,1,2,3,4,5 BDD

    #PAra o kitti as labels da gt sao :    34,5,1,2
    #As previsoes do yolo estao nos lugares 0,1,2,7
    #Ao escolher apenas estas vamos para:   0,1,2,3
    if kitti:
        cat_mapping_dict = {5: 1, 4: 0, 3: 0, 2: 7, 1: 2}
    else:
        cat_mapping_dict = {7: 3, 6: 1, 5: 0, 4: 0, 3: 7, 2: 5, 1: 2}
    with torch.no_grad():
        # Build preliminary dicts required for computing classification scores.
        for matched_results_key in matched_results.keys():
            if 'gt_cat_idxs' in matched_results[matched_results_key].keys():
                # First we convert the written things indices to contiguous
                # indices.
                #Esta secção serve para converter as labels das ground truth do bdd para yolo
                teste = matched_results[matched_results_key]['gt_cat_idxs']
                gt_converted_cat_idxs = matched_results[matched_results_key]['gt_cat_idxs'].squeeze(
                    1)
                #gt_converted_cat_idxs = matched_results[matched_results_key]['gt_cat_idxs']
                gt_converted_cat_idxs = torch.as_tensor([cat_mapping_dict[class_idx.cpu(
                ).tolist()] for class_idx in gt_converted_cat_idxs]).to(device)
                #convert from 0,1,2,3,5,7 to 0,1,2,3,4,5
                if kitti: #convert from 0,1,2,7 to 0,1,2,3
                    yolo_to_0_5_dict = {0: 0, 1: 1, 2: 2, 7: 3}
                else: #convert from 0,1,2,3,5,7 to 0,1,2,3,4,5
                    yolo_to_0_5_dict = {0: 0, 1: 1, 2: 2, 3: 3, 5: 4, 7: 5}
                gt_converted_cat_idxs = torch.as_tensor([yolo_to_0_5_dict[class_idx.cpu(
                ).tolist()] for class_idx in gt_converted_cat_idxs]).to(device)
                matched_results[matched_results_key]['gt_converted_cat_idxs'] = gt_converted_cat_idxs.to(
                    device)
                matched_results[matched_results_key]['gt_cat_idxs'] = gt_converted_cat_idxs
                #print(torch.unique(gt_converted_cat_idxs))
            if 'predicted_cls_probs' in matched_results[matched_results_key].keys(
            ):
                #Nesta secção pretende-se obter a classe prevista (aquela que possui o maior CS) para a bbox
                #Sera depois utilizado para fazer uma comparaçao com as GT e obter calibration errors
                #Guarda-se entao o CS score maximo obtido, e a classe correspondente.
                #Por alguma razão, no codigo original, ele retira a ultima coluna do vetor de probabilidades(motorcycle)
                #No nosso caso, queremos todas, por isso vamos deixar assim, ja fizemos um pre processemanto que garante todas as categorias relevantes
                #predicted_class_probs, predicted_cat_idxs = matched_results[
                #    matched_results_key]['predicted_cls_probs'][:, :-1].max(1)
                #Vamos fazer uma limpeza aos dados, ficando apenas as 6 categorias relevantes, removendo as outras 74
                #Esta parte afinal é necessaria porque precisamos disto para calcular os calibration errors, que apenas sao relevantes para estas classes
                #Neste momento as classes que serão predicted vao do 0 ate ao 5 inclusive
                #0-person/rider|1-bycicle|2-car|3-motorcycle|4-bus|5-truck
                matched_results[matched_results_key]['predicted_cls_probs'] = matched_results[matched_results_key]['predicted_cls_probs'][:,[0,1,2,3,5,7]]
                predicted_class_probs, predicted_cat_idxs = matched_results[
                    matched_results_key]['predicted_cls_probs'].max(1)
                #teste_probs, teste_idxs = teste.max(1)
                #print(torch.unique(predicted_cat_idxs))
                matched_results[matched_results_key]['predicted_cat_idxs'] = predicted_cat_idxs
                matched_results[matched_results_key]['output_logits'] = predicted_class_probs

        # Load the different detection partitions
        true_positives = matched_results['true_positives']
        duplicates = matched_results['duplicates']
        false_positives = matched_results['false_positives']

        # Get the number of elements in each partition
        cls_min_uncertainty_error_list = []

        reg_maximum_calibration_error_list = []
        reg_expected_calibration_error_list = []
        reg_min_uncertainty_error_list = []

        all_predicted_scores = torch.cat(
            (true_positives['predicted_cls_probs'].flatten(),
             duplicates['predicted_cls_probs'].flatten(),
             false_positives['predicted_cls_probs'].flatten()),
            0)

        all_gt_scores = torch.cat(
            (torch.nn.functional.one_hot(
                true_positives['gt_cat_idxs'],
                true_positives['predicted_cls_probs'].shape[1]).flatten().to(device),
                torch.nn.functional.one_hot(
                duplicates['gt_cat_idxs'],
                true_positives['predicted_cls_probs'].shape[1]).flatten().to(device),
                torch.zeros_like(
                false_positives['predicted_cls_probs'].type(
                    torch.LongTensor).flatten()).to(device)),
            0)

        # Compute classification calibration error using calibration
        # library
        #O marginal calibration error é para o caso de multi-classe
        cls_marginal_calibration_error = cal.get_calibration_error(
            all_predicted_scores.cpu().numpy(), all_gt_scores.cpu().numpy())
        
        #Nesta secção vamos classe a classe calcular regression calibration error, minimum uncertainty error tanto para cls como reg.
        #No fim, faz-se uma média de todas as classes
        #for class_idx in cat_mapping_dict.values():
        if kitti:
            categories_list = [0,1,2,3]
        else:
            categories_list = [0,1,2,3,4,5]

        for class_idx in categories_list:
            true_positives_valid_idxs = true_positives['gt_converted_cat_idxs'] == class_idx
            duplicates_valid_idxs = duplicates['gt_converted_cat_idxs'] == class_idx
            false_positives_valid_idxs = false_positives['predicted_cat_idxs'] == class_idx

            # For the rest of the code, gt_scores need to be ones or zeros. All
            # processing is done on a per-class basis
            #Como estamos a analisar classe a classe, basta ter 1 ou 0, é ou não é.
            all_gt_scores = torch.cat(
                (torch.ones_like(
                    true_positives['gt_converted_cat_idxs'][true_positives_valid_idxs]).to(device),
                    torch.zeros_like(
                    duplicates['gt_converted_cat_idxs'][duplicates_valid_idxs]).to(device),
                    torch.zeros_like(
                    false_positives['predicted_cat_idxs'][false_positives_valid_idxs]).to(device)),
                0).type(
                torch.DoubleTensor)

            # Compute classification minimum uncertainty error
            #Neste vetor vamos colocar as probabilidades obtidas(CS maximo) para a classe. Essencial para de seguida calcular a entropia
            distribution_params = torch.cat(
                (true_positives['output_logits'][true_positives_valid_idxs],
                 duplicates['output_logits'][duplicates_valid_idxs],
                 false_positives['output_logits'][false_positives_valid_idxs]),
                0)
            #Agora calculamos a entropia, atraves das probabilidades. Cada bbox vai ter a sua entropia associada a classe aqui.
            all_predicted_cat_entropy = -torch.log(distribution_params)
            #Returns a random permutation of integers from 0 to n - 1.
            #Nesta pequena secção ele realiza uma randomização dos valores.
            #Nao percebo a necessidade disto, ate porque a seguir vamos ordenar por entropia obtida.
            random_idxs = torch.randperm(all_predicted_cat_entropy.shape[0])
            all_predicted_cat_entropy = all_predicted_cat_entropy[random_idxs]
            all_gt_scores_cls = all_gt_scores[random_idxs]
            #Aqui vamos obter a ordem correta, começando por valores de entropia mais baixos para os maiores.
            sorted_entropies, sorted_idxs = all_predicted_cat_entropy.sort()
            #O nosso objetivo é contar quantos TP estao acima de um certo valor de entropia(e dividir pelo numero total de TP e multiplicar por 0.5)
            #Assim como contar quantos FP estao abaixo de um certo valor de entropia(e dividir pelo numero total de FP e multiplicar por 0.5)
            #Somando estes dois valores obtemos o uncertainty error.
            #Criar estes dois vetores "simetricos" da a possibilidade de contar o numero total de TP(basta fazer a soma do vetor, se for tp é 1, 0 otherwise)
            #Assim como o seu simetro permite a contagem do numero total de FP.
            sorted_gt_idxs_tp = all_gt_scores_cls[sorted_idxs]
            sorted_gt_idxs_fp = 1.0 - sorted_gt_idxs_tp
            #O cumulative sum serve para podermos experimentar todos os possiveis spots de colocar o threshold da entropia!
            #Começas por colocar apenas o primeiro valor mais pequeno de entropia, e vais subindo, obtens uma matriz com todos os valores de 
            #uncertainty error para cada possivel threshold. Depois basta procurar qual o valor minimo obtido e esta ai o Minimum Uncertainty Error
            tp_cum_sum = torch.cumsum(sorted_gt_idxs_tp, 0)
            fp_cum_sum = torch.cumsum(sorted_gt_idxs_fp, 0)
            #esta soma representa o numero total de ground truths que existem e por isso
            #o numero maximo de true positives possivel

            cls_u_errors = 0.5 * (sorted_gt_idxs_tp.sum(0) - tp_cum_sum) / \
                sorted_gt_idxs_tp.sum(0) + 0.5 * fp_cum_sum / sorted_gt_idxs_fp.sum(0)
            cls_min_u_error = cls_u_errors.min()
            cls_min_uncertainty_error_list.append(cls_min_u_error)

            # Compute regression calibration errors. False negatives cant be evaluated since
            # those do not have ground truth.
            #Obtemos os dados relevantes para calibration da regressao, medias e matrizes de covariancia(so vamos precisar das variancias)
            all_predicted_means = torch.cat(
                (true_positives['predicted_box_means'][true_positives_valid_idxs],
                 duplicates['predicted_box_means'][duplicates_valid_idxs]),
                0)

            all_predicted_covariances = torch.cat(
                (true_positives['predicted_box_covariances'][true_positives_valid_idxs],
                 duplicates['predicted_box_covariances'][duplicates_valid_idxs]),
                0)

            all_predicted_gt = torch.cat(
                (true_positives['gt_box_means'][true_positives_valid_idxs],
                 duplicates['gt_box_means'][duplicates_valid_idxs]),
                0)
            #Aqui retiramos apenas as variancias das 4 variaveis
            all_predicted_covariances = torch.diagonal(
                all_predicted_covariances, dim1=1, dim2=2)

            # The assumption of uncorrelated components is not accurate, especially when estimating full
            # covariance matrices. However, using scipy to compute multivariate cdfs is very very
            # time consuming for such large amounts of data.
            reg_maximum_calibration_error = []
            reg_expected_calibration_error = []

            # Regression calibration is computed for every box dimension
            # separately, and averaged after.
            for box_dim in range(all_predicted_gt.shape[1]):
                #Vamos obter os valores da calibraçao da regressao variavel a variavel da bbox, e apenas no fim se faz a average(tambem estamos a fazer apenas para uma classe)
                all_predicted_means_current_dim = all_predicted_means[:, box_dim]
                all_predicted_gt_current_dim = all_predicted_gt[:, box_dim]
                all_predicted_covariances_current_dim = all_predicted_covariances[:, box_dim]
                #Aqui criamos distribuiçoes normais para cada prediction usando a media e o desvio padrao(raiz quadrada da variancia)
                normal_dists = torch.distributions.Normal(
                    all_predicted_means_current_dim,
                    scale=torch.sqrt(all_predicted_covariances_current_dim))
                #Aqui obtemos o valor para a "likelihood" da predictive distribution estimada "explicar" o ground truth
                #Ou seja, a probabilidade do ground truth pertencer a esta distribution
                #Obtem se a probabilidade que "the variable will fall into a certain interval that you supply"
                all_predicted_scores = normal_dists.cdf(
                    all_predicted_gt_current_dim)

                reg_calibration_error = []
                histogram_bin_step_size = 1 / 15.0
                for i in torch.arange(
                        0.0,
                        1.0 - histogram_bin_step_size,
                        histogram_bin_step_size):
                    # Get number of elements in bin
                    #Aqui vamos criar varios bins para colocar as diferentes previsoes, para cada bin vamos fazer um calculo de quantas previsoes
                    #possuem um score/probabilidade abaixo daquele certo valor
                    #Com esse numero e dividindo pelo numero total de previsoes, conseguimos saber a "accuracy" obtida e podemos comparar com 
                    #a probabilidade expectada que é o valor maximo de confidence do bin. 
                    #Depois para calcular a calibration fazemos a diferença entre esses dois valores e elevamos ao quadrado.
                    #Por fim temos de fazer uma media para todos os bins obtidos, e ficamos assim com uma Expected calibration error
                    #E tambem obtemos a maximum claibration error. Nao esquecer que isto apenas representa uma das variaveis.
                    #É necessario fazer isto para todas as variaveis, depois fazer a media. Mesmo assim, apenas representa uma classe, fazer a media entre classes
                    elements_in_bin = (
                        all_predicted_scores < (i + histogram_bin_step_size))
                    num_elems_in_bin_i = elements_in_bin.type(
                        torch.FloatTensor).to(device).sum()

                    # Compute calibration error from "Accurate uncertainties for deep
                    # learning using calibrated regression" paper.
                    #teste = ((num_elems_in_bin_i / all_predicted_scores.shape[0]) - (i + histogram_bin_step_size)) ** 2
                    #igual = num_elems_in_bin_i / all_predicted_scores.shape[0]
                    #igual = (igual - (i + histogram_bin_step_size)) ** 2
                    #teste = (i + histogram_bin_step_size)
                    #teste = num_elems_in_bin_i / all_predicted_scores.shape[0] - (i + histogram_bin_step_size)
                    reg_calibration_error.append(
                        (num_elems_in_bin_i / all_predicted_scores.shape[0] - (i + histogram_bin_step_size)) ** 2)

                calibration_error = torch.stack(
                    reg_calibration_error).to(device)
                reg_maximum_calibration_error.append(calibration_error.max())
                reg_expected_calibration_error.append(calibration_error.mean())

            reg_maximum_calibration_error_list.append(
                reg_maximum_calibration_error)
            reg_expected_calibration_error_list.append(
                reg_expected_calibration_error)

            # Compute regression minimum uncertainty error
            #É completamente similar a minimum uncertainty error para a classificação
            #A diferença é que neste caso para obtermos os valores de entropia tem que ser atraves de distribuiçoes multivariadas
            #cuidado, é possivel termos de multiplicar por fatores para garantir positive definite matrices.
            all_predicted_covars = torch.cat(
                (true_positives['predicted_box_covariances'][true_positives_valid_idxs],
                 duplicates['predicted_box_covariances'][duplicates_valid_idxs],
                 false_positives['predicted_box_covariances'][false_positives_valid_idxs]),
                0)

            #Code(3 lines) that guarantee matrices to be positive definite
            diag_up = torch.triu(all_predicted_covars)
            upper = torch.triu(all_predicted_covars,diagonal=1)
            all_predicted_covars = torch.transpose(diag_up,1,2) + upper
            
            #count = 0
            #for i , cov_matrix in enumerate(all_predicted_covars):
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
               
            all_predicted_distributions = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(
                all_predicted_covars.shape[0:2]).to(device), all_predicted_covars + 1e-4 * torch.eye(all_predicted_covars.shape[2]).to(device))

            all_predicted_reg_entropy = all_predicted_distributions.entropy()
            random_idxs = torch.randperm(all_predicted_reg_entropy.shape[0])

            all_predicted_reg_entropy = all_predicted_reg_entropy[random_idxs]
            all_gt_scores_reg = all_gt_scores[random_idxs]

            sorted_entropies, sorted_idxs = all_predicted_reg_entropy.sort()
            sorted_gt_idxs_tp = all_gt_scores_reg[sorted_idxs]
            sorted_gt_idxs_fp = 1.0 - sorted_gt_idxs_tp

            tp_cum_sum = torch.cumsum(sorted_gt_idxs_tp, 0)
            fp_cum_sum = torch.cumsum(sorted_gt_idxs_fp, 0)
            reg_u_errors = 0.5 * ((sorted_gt_idxs_tp.sum(0) - tp_cum_sum) /
                                  sorted_gt_idxs_tp.sum(0)) + 0.5 * (fp_cum_sum / sorted_gt_idxs_fp.sum(0))
            reg_min_u_error = reg_u_errors.min()
            reg_min_uncertainty_error_list.append(reg_min_u_error)
            # Summarize and print all
        table = PrettyTable()
        table.field_names = (['Cls Marginal CE',
                              'Reg Expected CE',
                              'Reg Maximum CE',
                              'Cls MUE',
                              'Reg MUE'])

        reg_expected_calibration_error = torch.stack([torch.stack(
            reg, 0) for reg in reg_expected_calibration_error_list], 0)
        reg_expected_calibration_error = reg_expected_calibration_error[
            ~torch.isnan(reg_expected_calibration_error)].mean()

        reg_maximum_calibration_error = torch.stack([torch.stack(
            reg, 0) for reg in reg_maximum_calibration_error_list], 0)
        reg_maximum_calibration_error = reg_maximum_calibration_error[
            ~torch.isnan(reg_maximum_calibration_error)].mean()

        cls_min_u_error = torch.stack(cls_min_uncertainty_error_list, 0)
        cls_min_u_error = cls_min_u_error[
            ~torch.isnan(cls_min_u_error)].mean()

        reg_min_u_error = torch.stack(reg_min_uncertainty_error_list, 0)
        reg_min_u_error = reg_min_u_error[
            ~torch.isnan(reg_min_u_error)].mean()

        table.add_row(['{:.4f}'.format(cls_marginal_calibration_error),
                       '{:.4f}'.format(reg_expected_calibration_error.cpu().numpy().tolist()),
                       '{:.4f}'.format(reg_maximum_calibration_error.cpu().numpy().tolist()),
                       '{:.4f}'.format(cls_min_u_error.cpu().numpy().tolist()),
                       '{:.4f}'.format(reg_min_u_error.cpu().numpy().tolist())])
        print(table)
        final_results_calibration = {"cls_marginal_cal_error":cls_marginal_calibration_error,
                                     "reg_expected_cal_error":reg_expected_calibration_error,
                                     "reg_max_cal_error":reg_maximum_calibration_error,
                                     "cls_min_u_error":cls_min_u_error,
                                     "reg_min_u_error":reg_min_u_error}

    return final_results_calibration

def compute_average_precision(path_to_results,path_to_dataset,kitti = False):

    # Build path to inference output
    inference_output_dir = path_to_results

    prediction_file_name = os.path.join(
        inference_output_dir,
        'coco_instances_results_xywh.json')

    #meta_catalog = MetadataCatalog.get(args.test_dataset)
    meta_catalog_json_file = os.path.join(path_to_dataset,'val_coco_format.json') 
    # Evaluate detection results
    #gt_coco_api = COCO(meta_catalog.json_file)
    gt_coco_api = COCO(meta_catalog_json_file)
    res_coco_api = gt_coco_api.loadRes(prediction_file_name)
    results_api = COCOeval(gt_coco_api, res_coco_api, iouType='bbox')
    
    #Use this for mAP only with "car" and "person" as categories (for dataset shift mAP)
    #Eu retirei as labels dos "riders" aqui, pois nao da para simplesmente os colocar como se fossem da classe "person"
    #Problema atual: como eu nas deteçoes ponho tudo junto, vai haver riders que sao considerados persons e vao dar como falso positivos quando
    #eu nao considero os riders. Se eu considerar os riders tambem dao como falsos positivos pois a classe vai estar errada
    #Caso queira incluir a unica soluçao seria: alterar as gt em si na sua geraçao, transformando os riders em persons
    
    #YOLO | BDD | Kitti
    #2 car | 1 car | 1 car
    #5 bus | 2 bus | ----
    #7 truck| 3 truck | 2 truck
    #0 person| 4 person | 3 person
    #-1 rider| 5 rider | 4 rider
    #1 bycicle | 6 bike | 5 bycicle
    #3 motorcycle | 7 motor -----
    if kitti:
        results_api.params.catIds = [1,2,3,5] 
    else:
        results_api.params.catIds = [1,2,3,4,6,7] #This only works for BDD!!! For kitti dataset, you should use either the list or [1,2]
    
    #Use this for standard mAP across all existing categories
    #results_api.params.catIds = list(meta_catalog.thing_dataset_id_to_contiguous_id.keys())
    #print(meta_catalog)
    #print(results_api.params.catIds)
    
    # Calculate and print aggregate results
    results_api.evaluate()
    results_api.accumulate()
    results_api.summarize()

    # Compute optimal micro F1 score threshold. We compute the f1 score for
    # every class and score threshold. We then compute the score threshold that
    # maximizes the F-1 score of every class. The final score threshold is the average
    # over all classes.
    precisions = results_api.eval['precision'].mean(0)[:, :, 0, 2]
    recalls = np.expand_dims(results_api.params.recThrs, 1)
    f1_scores = 2*(precisions * recalls) / (precisions + recalls)
    optimal_f1_score = f1_scores.argmax(0)
    scores = results_api.eval['scores'].mean(0)[:, :, 0, 2]
    optimal_score_threshold = [scores[optimal_f1_score_i, i] for i, optimal_f1_score_i in enumerate(optimal_f1_score)]
    optimal_score_threshold = np.array(optimal_score_threshold)
    optimal_score_threshold = optimal_score_threshold[optimal_score_threshold != 0]
    optimal_score_threshold = optimal_score_threshold.mean()

    print("Classification Score at Optimal F-1 Score: {}".format(optimal_score_threshold))

    text_file_name = os.path.join(
        inference_output_dir,
        'mAP_res.txt')

    with open(text_file_name, "w") as text_file:
        print(results_api.stats.tolist() +
              [optimal_score_threshold, ], file=text_file)

    return results_api.stats.tolist(), optimal_score_threshold