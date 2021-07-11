from sklearn import svm

import numpy as np

from util import *


class LatentManipulator(Session):

    def __init__(self, session_id):

        super(LatentManipulator, self).__init__(session_id)

        try:
            self.config['boundaries']
        except KeyError:
            self.config['boundaries'] = {}

        self.boundaries = self.config['boundaries']
        self.semantics = self.config['semantics']

        data_path = root_dir + 'data/' + self.session_id + '_scores_' + '_'.join(self.semantics) + '.npz'
        data = np.load(data_path)
        latents = data['latents']
        scores = data['scores']

        for i in range(len(self.semantics)):
            self.create_boundary(self.semantics[i], latents, scores[:, i])

    def create_boundary(self, semantic, latents, scores, val_split=0.2):

        # Sort data wrt scores
        idx_s = np.argsort(scores)
        latents_sorted = latents[idx_s]
        scores_sorted = scores[idx_s]

        # Evenly split scores across origin
        scores_sorted -= np.median(scores_sorted)
        labels = np.sign(scores_sorted)

        # Find boundary
        print('Fitting new boundary for', semantic)
        split_idx = int(len(latents) * (1 - val_split))
        model = svm.SVC(kernel='linear')
        model.fit(latents_sorted[:split_idx], labels[:split_idx])
        boundary = model.coef_.reshape(1, latents_sorted.shape[1]).astype(np.float32)
        boundary / np.linalg.norm(boundary)

        # Summary
        acc = np.mean(np.abs(model.predict(latents_sorted[split_idx:]) + labels[split_idx:]) / 2)

        self.boundaries[semantic] = boundary
        print('--Boundary added with validation accuracy', acc)

    def move_latent(self, latent, semantic, alpha):
        return latent + alpha * self.boundaries[semantic]

    def move_latent_conditional(self):

        pass


if __name__ == '__main__':

    lm = LatentManipulator('pgf1')
