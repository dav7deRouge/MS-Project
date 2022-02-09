import numpy as np
import numpy.random as rnd
import math
from sklearn.gaussian_process.kernels import RBF
from scipy.spatial.distance import cdist
import numpy.matlib as ml
import scipy.linalg as la
from sklearn import neighbors
import scipy.io as spio

def sca(X_source, Y_source, X_t, Y_t, params = { 'beta': [0.1, 0.3, 0.5, 0.7, 0.9],
                                                 'delta': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6],
                                                 'k_list': [2],
                                                 'X_v': None,
                                                 'Y_v': None }):
  """ La funzione implementa l'algoritmo di Scatter Component Analysis. 
      X_source è un vettore di matrici, ogni matrice rappresenta un dominio.
      Y_source è un vettore di vettori di etichette, ogni vettore interno rappresenta un dominio.
      Si richiede di passare i parametri (params) sotto forma di dizionario, in caso contrario
      saranno utilizzati i parametri di default. """

  # Controllo che i parametri siano passati sotto forma di dizionario con i nomi corretti, altrimenti si usano quelli di default
  if type(params) != dict or params.get('beta') == None or params.get('delta') == None or params.get('k_list') == None:
    params = {  'beta': [0.1, 0.3, 0.5, 0.7, 0.9],
                'delta': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6],
                'k_list': [2],
                'X_v': X_source,
                'Y_v': Y_source }
  # Se si utilizzano i parametri di default è necessario utilizzare i source come validation sets:
  if (params.get('X_v').all() == None) or (params.get('Y_v').all() == None):
    params['X_v'] = X_source
    params['Y_v'] = Y_source

  beta = params.get('beta'); delta = params.get('delta'); k_list = params.get('k_list')

  # Unisco gli elementi dei vari domini in un solo vettore:
  X_s = np.asarray(X_source) # questa riga potrebbe non servire se X_source è già un array in numpy
  X_s = np.reshape(X_s, (X_s.shape[0]*X_s.shape[1], X_s.shape[2]))
  Y_s = np.asarray(Y_source)
  Y_s = np.reshape(Y_s, (Y_s.shape[0]*Y_s.shape[1]))

  print(f'Numero dei domini sorgente = {len(X_source)}, numero di classi = {len(np.unique(Y_s))}')

  dist_s_s = cdist(X_s, X_s)
  dist_s_s = dist_s_s ** 2
  sgm_s = lengthscale(dist_s_s)

  dist_s_v = cdist(X_s, X_v)
  dist_s_v = dist_s_v ** 2
  sgm_v = lengthscale(dist_s_v) # nel codice in matlab viene passato ancora dist_s_s, CONTROLLARE IN CASO DI ERRORE

  n_s = len(X_s)
  n_v = len(X_v)
  H_s = np.eye(n_s) - np.ones((n_s, n_s))/n_s
  H_v = np.eye(n_v) - np.ones((n_v, n_v))/n_v

  kernel_ss = RBF(length_scale=sgm_s)
  kernel_sv = RBF(length_scale=sgm_v)
  K_s_s = kernel_ss(X_s, X_s)
  K_s_v = kernel_sv(X_s, X_v)
  K_s_v_centered = np.matmul(np.matmul(H_s, K_s_v), H_v)
  P, T, D, Q, K_s_s_centered = sca_matrices(K_s_s, X_source, Y_source)

  acc_matrix = np.zeros((len(k_list), len(beta), len(delta)))
  for curr_beta in beta:
    for curr_delta in delta:
      B, A = sca_transformation(P, D, T, Q, K_s_s_centered, curr_beta, curr_delta, 1e-5)
      for k in k_list:
        accuracy = sca_test(B, A, K_s_s_centered, K_s_v_centered, Y_s, params.get('Y_v'), k)[0]
        acc_matrix[k_list.index(k), beta.index(curr_beta), delta.index(curr_delta)] = accuracy
        print(f'beta = {curr_beta}, delta = {curr_delta}, accuracy = {accuracy}')

  print(f'\nValidation done! Classifying the target domain instances ...')
  # Fase di testing:
  best_accuracy = np.amax(acc_matrix)
  ind = np.argwhere(acc_matrix == best_accuracy)
  best_beta = beta[ind[0][1]]; best_delta = delta[ind[0][2]]; best_k = k_list[ind[0][0]]

  print(f"best beta = {best_beta}, best delta = {best_delta}, best k = {best_k}")

  dist_s_t = cdist(X_s, X_t)
  dist_s_t = dist_s_t ** 2
  sgm = lengthscale(dist_s_t)
  kernel_st = RBF(length_scale=sgm)
  K_s_t = kernel_st(X_s, X_t)
  H_s = np.eye(X_s.shape[0]) - np.ones((X_s.shape[0], X_s.shape[0]))/X_s.shape[0]
  H_t = np.eye(X_t.shape[0]) - np.ones((X_t.shape[0], X_t.shape[0]))/X_t.shape[0]
  K_s_t_centered = np.matmul(np.matmul(H_s, K_s_t), H_t)

  B, A = sca_transformation(P, D, T, Q, K_s_s_centered, best_beta, best_delta, 1e-5)
  test_acc, pre_labels, Z_s, Z_t = sca_test(B, A, K_s_s_centered, K_s_t_centered, Y_s, Y_t, best_k)
  print(Z_s.shape, Z_t.shape)
  print(f'\nTest accuracy = {test_acc}')  

  return test_acc, pre_labels, Z_s, Z_t

def sca_transformation(P, D, T, Q, kernel_centered, beta, delta, epsilon):
  """ La funzione calcola la matrice di trasformazione B come richiesto dall'algoritmo. """
  I_0 = np.eye(kernel_centered.shape[0])
  F1 = (beta * P) + (((1 - beta)/P.shape[0]) * T)
  F2 = (delta * D) + Q + kernel_centered + (epsilon*I_0)
  F = la.solve(F2, F1) 
    
  A, B = la.eig(F)
  B = np.real(B)
  A = np.real(A)
  
  val = np.sort(A)[::-1]
  idx = np.argsort(A)[::-1]
  B = B[:, idx]
  A = np.diag(val)

  return B, A

def sca_matrices(kernel, source, labels):
  """ La funzione restituisce tutte le matrici necessarie per l'esecuzione dell'algoritmo SCA 
      Source è un vettore di matrici, ogni matrice rappresenta un dominio. 
      Quindi: source.shape[0] = numero di domini;
              source.shape[1] = numero di elementi per ciascun dominio;
              source.shape[2] = numero di features per ciascun elemento.
      Labels è un vettore di vettori di etichette, ogni vettore interno rappresenta un dominio """
  domains_num = len(source)

  total_elems = 0
  for dom in source:
    total_elems += len(dom)
  
  # Raggruppo le etichette in un solo vettore:
  all_labels = np.reshape(labels, (total_elems, 1))
  class_num = len(np.unique(all_labels))
  # Costruisco l'indice dei domini: 
  #(necessario perché la kernel matrix è costruita su tutti gli elementi indifferentemente dal dominio)
  domain_index = []
  i = 0
  for dom in source:
    for elem in dom:
      domain_index.append(i)
    i += 1
  domain_index = np.asarray(domain_index)
  # Costruisco l'indice delle classi:
  class_index = []
  for dom in labels:
    for elem in dom:
      class_index.append(elem)
  class_index = np.asarray(class_index)
  class_index = np.reshape(class_index, (1, class_index.shape[0]))
  # Costruisco un dizionario classe-elemento per facilitare il calcolo della matrice P:
  dictionary = {i: [] for i in range(1, class_num+1)}
  j = 0
  for dom in labels:
    i = 0
    for elem in dom:
      dictionary[elem].append(source[j][i])
      i += 1
    j += 1

  # Matrice indotta dal domain scatter:
  D = np.zeros((total_elems, total_elems))
  temp = np.zeros(total_elems)
  for dom in range(0, domains_num):
    temp = temp + np.mean(kernel[:, np.where(domain_index == dom)[0]], axis=1)
  temp = temp / domains_num
  for dom in range(0, domains_num):
    D[:, dom] = np.mean(kernel[:, np.where(domain_index == dom)[0]], axis=1) - temp
  D = np.matmul(D, D.T) / domains_num

  I = np.ones((total_elems, total_elems))*(1/total_elems)
  kernel_centered = kernel - np.matmul(I, kernel) - np.matmul(kernel, I) + np.matmul(np.matmul(I, kernel), I)

  # Matrice indotta dal total scatter:
  T = np.matmul(kernel_centered, kernel_centered.T) / total_elems

  # Matrice Q per il calcolo del within-class scatter:
  Q = np.zeros((total_elems, total_elems))
  for i in range(1, class_num+1):
    idx = np.where(class_index == i)[1]
    G_j = np.mean(kernel[:, idx], axis=1); G_j = np.reshape(G_j, (len(G_j), 1))
    
    G_ij = kernel[:, idx] 
    Q_i = G_ij - ml.repmat(G_j, 1, len(idx))
    Q = Q + np.matmul(Q_i, Q_i.T)

  # Matrice P per il calcolo del between-class scatter:
  P = np.zeros((total_elems, total_elems))
  class_idx = np.zeros((total_elems, 1), dtype=int)

  count = 0
  for i in dictionary.keys():
    for j in range(0, len(dictionary.get(i))):
      class_idx[count] = i
      count += 1

  P_mean = np.mean(kernel, axis=1) #centroide dei centroidi delle classi
  for j in range(0, total_elems):
    class_id = class_idx[j][0]
    temp_k = np.mean(kernel[:, np.where(class_index == class_id)[1]], axis=1); 
    P[:, j] = temp_k - P_mean
  P = np.matmul(P, P.T)

  return P, T, D, Q, kernel_centered

def sca_test(B, A, K_s, K_t, Y_s, Y_t, eig_ratio):
  """ Funzione che applica la trasformazione agli elementi appartenenti al dominio target """
  vals = np.diag(A)
  vals.setflags(write=1)

  ratio = []
  count = 0
  for i in range(0, len(vals)):
    if vals[i] < 0:
      break
    count += vals[i]
    ratio.append(count)
    vals[i] = 1 / math.sqrt(vals[i])
  A_sqrt = np.diag(vals)

  ratio = np.asarray(ratio)
  ratio = ratio / count
  if eig_ratio <= 1:
    idx = np.where(ratio > eig_ratio)[0]
    n_eigs = idx[0]
  else:
    n_eigs = eig_ratio

  Z_t = np.matmul(np.matmul(K_t.T, B[:, 1:n_eigs]), A_sqrt[1:n_eigs, 1:n_eigs])
  Z_s = np.matmul(np.matmul(K_s.T, B[:, 1:n_eigs]), A_sqrt[1:n_eigs, 1:n_eigs])

  # 11-NN classificatore:
  neighs = 11
  classifier = neighbors.KNeighborsClassifier(neighs, weights='distance')
  classifier.fit(Z_s, Y_s)
  pre_labels = classifier.predict(Z_t)
  accuracy = len(np.where(pre_labels == Y_t)[0]) / len(pre_labels)

  return accuracy, pre_labels, Z_s, Z_t

def lengthscale(distance_matrix):
  """ Funzione che calcola il parametro length_scale del kernel RBF come
      length_scale = median(|| a - b || ^ 2), per ogni a,b appartenenti a S^s U S^t """
  dim = distance_matrix.shape[0] * distance_matrix.shape[1]
  t = np.tril(distance_matrix)
  t = np.reshape(t, 1 * dim)
  t = t[t > 0]

  return math.sqrt(0.5 * np.median(t))