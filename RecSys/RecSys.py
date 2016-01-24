import numpy as np
import matplotlib.pyplot as plt
R = np.loadtxt("./q1-dataset/q1-dataset/user-shows.txt")
Q = np.diag(1/np.sqrt(R.sum(axis=0)))
P = np.diag(1/np.sqrt(R.sum(axis=1)))
S_iter = Q.dot(R.transpose()).dot(R.dot(Q))
S_user = P.dot(R).dot(R.transpose().dot(P))
rec_user = S_user[499,:].dot(R)
rec_iter = R[499,:].dot(S_iter)

f = open("./q1-dataset/q1-dataset/shows.txt")
shows = []
for line in f:
	shows.extend([line[1:-2]])

ind_user = np.argpartition(rec_user[0:100], -5)[-5:]
ind_iter = np.argpartition(rec_iter[0:100], -5)[-5:]
f = open('first_5_rec.txt', 'w+')
f.write("from user-user rec\n")
f.write(str([shows[i] for i in ind_user]))
f.write("\n")
f.write(str(rec_user[ind_user]))
f.write("\n")
f.write("from iter-iter rec\n")
f.write(str([shows[i] for i in ind_iter]))
f.write("\n")
f.write(str(rec_iter[ind_iter]))
f.write("\n")
f.close()

true = np.loadtxt("./q1-dataset/q1-dataset/alex.txt")[0:100]

#compute true positive for user-user rec
first_n = np.zeros(100)
count = sum(true)
rec = rec_user[0:100]
pos_user = np.zeros(19)
for k in range(0,19):
	first_n[rec.argmax()] = 1
	rec[rec.argmax()] = 0
	pos_user[k] = first_n.dot(true)/count

first_n = np.zeros(100)
rec = rec_iter[0:100]
pos_iter = np.zeros(19)
for k in range(0,19):
	first_n[rec.argmax()] = 1
	rec[rec.argmax()] = 0
	pos_iter[k] = first_n.dot(true)/count

l1, = plt.plot(range(1,20), pos_iter, label='iter rec positive rate')
l2, = plt.plot(range(1,20), pos_user, label='user rec positive rate')
plt.legend([l1,l2], ['iter rec positive rate', 'user rec positive rate'], loc=2)
plt.savefig('positive_rate')