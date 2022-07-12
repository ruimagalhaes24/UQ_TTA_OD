import torch
from torchvision.ops import batched_nms
#Detectron imports 
#Possiveis soluções: restart connection; install versoes anteriores; clone do repositorio localmente; escrever funçao iou
from detectron2.detectron2.structures import BoxMode, Boxes, pairwise_iou, Instances

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#This function will grab the prediction output from yolo (1,x,85) and separate this information into different pieces
#relevant to the output redundancy method
#LIST THE PIECES HERE
def pre_processing_anchor_stats(pred_,max_number_bboxes = 20000):
    
    #Choose only the bbox predictions that assume the presence of an object
    #This is relevant because the training of the network disregards optimization of the bbox coordinates if there isn't an object present.
    #THE FIFTH VALUE IS THE 0/1 PRESENCE OR NOT OF AN OBJECT
    presence_of_object = torch.squeeze(pred_,dim=0)[:,5]
    relevant_prediction = presence_of_object > 0.5
    #obtain list with only predicted boxes and its coordinates. Also only the ones where there is the possibility of an object
    predicted_boxes = torch.squeeze(pred_,dim=0)[:,0:4][relevant_prediction]
    #obtain list with the probability scores for each class for each bbox
    predicted_prob_vectors = torch.squeeze(pred_,dim=0)[:,5:][relevant_prediction]
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
    predicted_boxes_converted[:,2] = predicted_boxes[:,0] + predicted_boxes[:,2]/2 #xmax = x + w/2
    #Atenção aos sinais, isto depende do referencial que estas a considerar. Há quem coloque a origem do referencial em cima à esquerda da bbox
    #e ha quem coloque o referencial em baixo à esquerda do referencial.
    #neste caso o eixo vertical esta reversed, o que implica uma mudança de sinal no y
    #Origem em bottom/left
    #predicted_boxes_converted[:,1] = predicted_boxes[:,1] + predicted_boxes[:,3]/2 #ymin = y + h/2
    #predicted_boxes_converted[:,3] = predicted_boxes[:,1] - predicted_boxes[:,3]/2 #ymax = y - h/2
    #Origem em top/left
    predicted_boxes_converted[:,1] = predicted_boxes[:,1] - predicted_boxes[:,3]/2 #ymin = y - h/2
    predicted_boxes_converted[:,3] = predicted_boxes[:,1] + predicted_boxes[:,3]/2 #ymax = y + h/2

    return predicted_boxes_converted, predicted_boxes_covariance, predicted_prob, classes_idxs, predicted_prob_vectors

def compute_anchor_statistics(outputs, device, image_size,
                                nms_threshold = 0.5, max_detections_per_image = 20000,affinity_threshold = 0.9):
        
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

def probabilistic_detector_postprocessing(outputs, image_size):
    """
    Resize the output instances and scales estimated covariance matrices.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.

    Args:
        results (Dict): the raw outputs from the probabilistic detector.
            `results.image_size` contains the input image resolution the detector sees.
            This object might be modified in-place.
        output_height: the desired output resolution.
        output_width: the desired output resolution.

    Returns:
        results (Dict): dictionary updated with rescaled boxes and covariance matrices.
    """
    #Este passo provavelmente é desnecessário visto que ambos os sizes da imagem são iguais
    #Neste caso do yolo e como fiz o codigo
    #No retinanet, ele tem um size para o tamanho original da imagem (que é suposto ser 1280x720)
    #e outro size quando efetivamente lê a imagem e avalia as matrizes (3,750,1333) pex
    #Vou deixar estar caso falte algo, either way simplesmente nao vai alterar nada.
    output_width = image_size.shape[1]
    output_height = image_size.shape[0]
    scale_x, scale_y = (output_width /
                        outputs.image_size[1], output_height /
                        outputs.image_size[0])

    outputs = Instances((output_height, output_width), **outputs.get_fields())

    output_boxes = outputs.pred_boxes
    # Scale bounding boxes
    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(outputs.image_size)
    #ATENÇAO ESTE .NONEMPTY FAZ COM QUE APENAS SOBREM 2 BBOXES FOR SOME REASON
    #non empty method faz xmax - xmin e ymax - ymin. se estas distancias
    #forem maior que 0, entao existe caixa.
    non_empty = output_boxes.nonempty()
    outputs = outputs[output_boxes.nonempty()]

    # Scale covariance matrices
    if outputs.has("pred_boxes_covariance"):
        # Add small value to make sure covariance matrix is well conditioned
        output_boxes_covariance = outputs.pred_boxes_covariance + 1e-4 * torch.eye(outputs.pred_boxes_covariance.shape[2]).to(device)

        scale_mat = torch.diag_embed(
            torch.as_tensor(
                (scale_x,
                 scale_y,
                 scale_x,
                 scale_y))).to(device).unsqueeze(0)
        scale_mat = torch.repeat_interleave(
            scale_mat, output_boxes_covariance.shape[0], 0)
        output_boxes_covariance = torch.matmul(
            torch.matmul(
                scale_mat,
                output_boxes_covariance),
            torch.transpose(scale_mat, 2, 1))
        outputs.pred_boxes_covariance = output_boxes_covariance
    return outputs

def covar_xyxy_to_xywh(output_boxes_covariance):
    """
    Converts covariance matrices from top-left bottom-right corner representation to top-left corner
    and width-height representation.

    Args:
        output_boxes_covariance: Input covariance matrices.

    Returns:
        output_boxes_covariance (Nxkxk): Transformed covariance matrices
    """
    transformation_mat = torch.as_tensor([[1.0, 0, 0, 0],
                                          [0, 1.0, 0, 0],
                                          [-1.0, 0, 1.0, 0],
                                          [0, -1.0, 0, 1.0]]).to(device).unsqueeze(0)
    transformation_mat = torch.repeat_interleave(
        transformation_mat, output_boxes_covariance.shape[0], 0)
    output_boxes_covariance = torch.matmul(
        torch.matmul(
            transformation_mat,
            output_boxes_covariance),
        torch.transpose(transformation_mat, 2, 1))

    return output_boxes_covariance

def instances_to_json(instances,img_id):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances): detectron2 instances
        img_id (int): the image id
        cat_mapping_dict (dict): dictionary to map between raw category id from net and dataset id. very important if
        performing inference on different dataset than that used for training.

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []
    #FOR BDD ISTO ESTA ERRADO, TENHO DE FAZER A CONVERSAO
    #YOLO PARA BDD? ver classes do coco no coco.yaml
    #YOLO: 0: Person; 1: Bycicle, 2: Car    
    #BDD 1: pedestrian 2: rider 
    # 3: car 4: truck | 5: bus|6: train|7: motorcycle|8: bicycle|9: traffic light
    # 10: traffic sign    
    #VER CONVERT BDD TO COCO
    #categories = [{'id': 1, 'name': 'car', 'supercategory': 'vehicle'},
    #              {'id': 2, 'name': 'bus', 'supercategory': 'vehicle'},
    #              {'id': 3, 'name': 'truck', 'supercategory': 'vehicle'},
    #              {'id': 4, 'name': 'person', 'supercategory': 'vehicle'},
    #              {'id': 5, 'name': 'rider', 'supercategory': 'vehicle'},
    #              {'id': 6, 'name': 'bike', 'supercategory': 'vehicle'},
    #              {'id': 7, 'name': 'motor', 'supercategory': 'vehicle'}
    #              ]
    #YOLO | BDD
    #2 car | 1 car
    #5 bus | 2 bus
    #7 truck| 3 truck
    #0 person| 4 person
    #-1 rider| 5 rider
    #1 bycicle | 6 bike 
    #3 motorcycle | 7 motor
    
    #IMPORTANTE, ESTOU A FICAR COM POUCAS BBOXES FINAIS (13 DE 100 NA PRIMEIRA IMAGEM)
    #ISTO ACONTECE PORQUE GRANDE PARTE DAS PREVISOES SAO DE OUTRAS CLASSES (STREET LIGHTS, SIGNS,ETC)
    #PODE SER RELEVANTE FAZER UM PRE PROCESSAMENTO ONDE NA PARTE DA OUTPUT REDUNDANCY SO ESCOLHO
    #AS 100 MELHORES BBOXES DAS CLASSES QUE ME INTERESSAM!!!
    cat_mapping_dict = {2: 1, 5: 2, 7: 3, 0: 4, 1: 6, 3: 4}

    boxes = instances.pred_boxes.tensor.cpu().numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.cpu().tolist()
    classes = instances.pred_classes.cpu().tolist()

    classes = [
        cat_mapping_dict[class_i] if class_i in cat_mapping_dict.keys() else -
        1 for class_i in classes]

    pred_cls_probs = instances.pred_cls_probs.cpu().tolist()

    if instances.has("pred_boxes_covariance"):
        pred_boxes_covariance = covar_xyxy_to_xywh(
            instances.pred_boxes_covariance).cpu().tolist()
    else:
        pred_boxes_covariance = []

    results = []
    for k in range(num_instance):
        if classes[k] != -1:
            result = {
                "image_id": img_id,
                "category_id": classes[k],
                "bbox": boxes[k],
                "score": scores[k],
                "cls_prob": pred_cls_probs[k],
                "bbox_covar": pred_boxes_covariance[k]
            }

            results.append(result)
    return results