import torch
from torchvision.ops import batched_nms
#Detectron imports 
#Possiveis soluções: restart connection; install versoes anteriores; clone do repositorio localmente; escrever funçao iou
from detectron2.detectron2.structures import Boxes, pairwise_iou, Instances

import gc

#This function will grab the prediction output from yolo (1,x,85) and separate this information into different pieces
#relevant to the output redundancy method
#LIST THE PIECES HERE
def pre_processing_anchor_stats(pred_,max_number_bboxes = 1000):
    
    #temp_pred = torch.squeeze(pred_,dim=0)
    #obtain list with only predicted boxes and its coordinates
    predicted_boxes = torch.squeeze(pred_,dim=0)[:,0:4]
    #THE FIFTH VALUE IS THE 0/1 PRESENCE OR NOT OF AN OBJECT
    #obtain list with the probability scores for each class for each bbox
    #presence_of_object = temp_pred[:,5]
    predicted_prob_vectors = torch.squeeze(pred_,dim=0)[:,5:]
    #find the highest confidence score for each bbox and class
    predicted_prob, classes_idxs = torch.max(predicted_prob_vectors, 1)
    #sort list of bboxes by CS (highest to lowest)
    predicted_prob, sorted_idxs = torch.sort(predicted_prob, descending = True)
    #sort all other lists to match the order obtained previously (highest to lowest CS)
    predicted_boxes = predicted_boxes[sorted_idxs]
    predicted_boxes_covariance = []
    #predicted_prob = predicted_prob
    classes_idxs = classes_idxs[sorted_idxs]
    predicted_prob_vectors = predicted_prob_vectors[sorted_idxs]
    
    #to try and avoid the memory problem (18900 was too much) 16750 seems to be the maximum
    #Pelo quue percebi, isto simplesmente tem a ver com a memory usage do gpu. Se outra pessoa estiver a correr outro programa
    #provavelmente nao vou conseguir utilizar tantas bboxes.
    #FAZER ESTUDO SOBRE QUANTAS BBOXES O RETINANET USA. a volta de 3900, NAO PASSOU DAS 4000!
    #PROCURAR POR PRE PROCESSAMENTO, COMO E QUE ELE ESCOLHE ES
    #Lets try with fucntions only later, maybe it works.
    predicted_boxes = predicted_boxes[:max_number_bboxes]
    predicted_prob = predicted_prob[:max_number_bboxes]
    classes_idxs = classes_idxs[:max_number_bboxes]
    predicted_boxes_covariance = []
    predicted_prob_vectors = predicted_prob_vectors[:max_number_bboxes]

    #Antes do iou é necessario converter [x,y,w,h] para [xmin,ymin,xmax,ymax]
    predicted_boxes_converted = predicted_boxes.clone()
    predicted_boxes_converted[:,0] = predicted_boxes[:,0] - predicted_boxes[:,2]/2 #xmin = x - w/2
    predicted_boxes_converted[:,1] = predicted_boxes[:,0] + predicted_boxes[:,2]/2 #xmax = x + w/2
    predicted_boxes_converted[:,2] = predicted_boxes[:,1] - predicted_boxes[:,3]/2 #ymin = x - h/2
    predicted_boxes_converted[:,3] = predicted_boxes[:,1] + predicted_boxes[:,3]/2 #ymin = x + h/2

    return predicted_boxes_converted, predicted_boxes_covariance, predicted_prob, classes_idxs, predicted_prob_vectors

def compute_anchor_statistics(outputs, device, image_size,
                                nms_threshold = 0.5, max_detections_per_image = 100,affinity_threshold = 0.9):
        
    predicted_boxes, predicted_boxes_covariance, predicted_prob, classes_idxs, predicted_prob_vectors = outputs
    # Get cluster centers using standard nms. Much faster than sequential
    # clustering.
    #keep vai possuir os indices das bbox com CS mais altos, por ordem,
    keep = batched_nms(
        predicted_boxes,
        predicted_prob,
        classes_idxs,
        nms_threshold)

    #aqui estamos apenas a limitar o numero maximo de bboxes
    keep = keep[: max_detections_per_image]

    #for i in range(81):
    #    print(predicted_prob_vectors[0][i])
    # Get pairwise iou matrix
    #obtem matriz MxN com iou entre boxes (diagonal é 1 e é simetrica)
    match_quality_matrix = pairwise_iou(Boxes(predicted_boxes), Boxes(predicted_boxes))

    #seleciona os ious das bbox que sobraram
    #fica uma matriz com 100 bbox(as melhores) por 1957 bbox(todas as outras possibilidades)
    clusters_inds = match_quality_matrix[keep, :]
    #filtra as bbox que possuem iou acima de um certo valor
    #por linha esta matriz vai te dizer quais sao as bbox que partilham bastante o espaço
    #com essa bbox da linha, que vai ser o cluster center
    clusters_inds = clusters_inds > affinity_threshold

    # Compute mean and covariance for every cluster.
    predicted_prob_vectors_list = []
    predicted_boxes_list = []
    predicted_boxes_covariance_list = []
    #o cluster_idxs vai buscar linha a linha da matriz 
    #ou seja, a lista das bbox que estao acima do affinity_threshold
    #o center_idx vai buscar o indice da bbox que vai ser o centro do cluster
    for cluster_idxs, center_idx in zip(
            clusters_inds, keep):

        if cluster_idxs.sum(0) >= 2: #se houver pelo menos mais uma bbox naquela zona
            # Make sure to only select cluster members of same class as center
            cluster_center_classes_idx = classes_idxs[center_idx] #vai buscar a classe atribuida À bbox do cluster center
            cluster_classes_idxs = classes_idxs[cluster_idxs] #vai buscar as classes atribuidas às bboxs do cluster
            class_similarity_idxs = cluster_classes_idxs == cluster_center_classes_idx 
            #lista que compara a classe do cluster com a classe das bboxes do cluster (T/F consoante se é igual ou nao)

            # Grab cluster
            box_cluster = predicted_boxes[cluster_idxs,:][class_similarity_idxs, :]
            #O QUE É QUE O CLASS_SIMILARITY_IDXS FAZ A ESTA MATRIZ??? essa variavel e que vai apenas selecionar as
            #bbox da matriz! o cluster:idxs tem todas as bbox la 1957
            cluster_mean = box_cluster.mean(0)
            #o box cluster fica assim uma matriz 49x4, 49 o numero de bbox e 4 que sao as variaveis da bbox
            #calcular a media dos valores da regressao.

            residuals = (box_cluster - cluster_mean).unsqueeze(2)
            #residuals para o calculo da variancia (valor do ponto- media)
            #formula da matriz da covariancia
            cluster_covariance = torch.sum(torch.matmul(residuals, torch.transpose(
                residuals, 2, 1)), 0) / max((box_cluster.shape[0] - 1), 1.0)

            # Assume final result as mean and covariance of gaussian mixture of cluster members if covariance is provided
            # by neural network
            if predicted_boxes_covariance is not None:
                if len(predicted_boxes_covariance) > 0:
                    cluster_covariance = cluster_covariance + \
                        predicted_boxes_covariance[cluster_idxs, :][class_similarity_idxs, :].mean(0)

            # Compute average over cluster probabilities
            #pegar em todos membros do cluster e fazer a media dos valores da prob para cada classe
            cluster_probs_vector = predicted_prob_vectors[cluster_idxs, :][class_similarity_idxs, :].mean(0)
        else: #so entras aqui caso so haja uma bbox que é o cluster center
            cluster_mean = predicted_boxes[center_idx]
            cluster_probs_vector = predicted_prob_vectors[center_idx]
            cluster_covariance = 1e-4 * torch.eye(4, 4).to(device)
            if predicted_boxes_covariance is not None:
                if len(predicted_boxes_covariance) > 0:
                    cluster_covariance = predicted_boxes_covariance[center_idx]

        predicted_boxes_list.append(cluster_mean) #lista que vai conter a media de cada cluster 
        predicted_boxes_covariance_list.append(cluster_covariance) #lista que vai conter a cov de cada cluster
        predicted_prob_vectors_list.append(cluster_probs_vector) #list que vai conter a media das classes de cada cluster
        
    result = Instances((image_size.shape[0],image_size.shape[1]))

    if len(predicted_boxes_list) > 0: #se houver mais do que um cluster 
        # We do not average the probability vectors for this post processing method. Averaging results in
        # very low mAP due to mixing with low scoring detection instances.
        result.pred_boxes = Boxes(torch.stack(predicted_boxes_list, 0)) #lista com os cluster centers e as suas medias 100,4
        predicted_prob_vectors = torch.stack(predicted_prob_vectors_list, 0)
        predicted_prob, classes_idxs = torch.max(
            predicted_prob_vectors, 1)
        result.scores = predicted_prob  #valor do CS para a melhor classe
        result.pred_classes = classes_idxs #classe prevista para o cluster center(a class com maior CS)
        result.pred_cls_probs = predicted_prob_vectors #lista as probs de cada classe para cada cluster center 100,7
        result.pred_boxes_covariance = torch.stack( 
            predicted_boxes_covariance_list, 0) ##lista com a matriz cov para cada cluster center 100, 4,4
    else:
        result.pred_boxes = Boxes(predicted_boxes)
        result.scores = torch.zeros(predicted_boxes.shape[0]).to(device)
        result.pred_classes = classes_idxs
        result.pred_cls_probs = predicted_prob_vectors
        result.pred_boxes_covariance = torch.empty(
            (predicted_boxes.shape + (4,))).to(device)
    return result




    
    #return predicted_boxes_list, predicted_boxes_covariance_list, predicted_prob_vectors_list