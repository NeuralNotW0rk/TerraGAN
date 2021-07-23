from sklearn import svm

import numpy as np

from util import *
from model import *


class LatentManipulator(Session):

    def __init__(self, session_id, version):

        super(LatentManipulator, self).__init__(session_id)

        path = root_dir + 'models/' + session_id + '/boundaries/'
        if not os.path.exists(path):
            os.mkdir(path)
        path += version + '.npz'

        if os.path.exists(path):
            self.boundaries = dict(np.load(path))
        else:
            self.boundaries = {}

        self.path = path

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

    def process_scores(self, n_latents=10000, from_file=False):

        self.boundaries = {}

        # Generate scores from model
        pgg = PGGAN(latent_size=self.config['latent_size'],
                    channels=self.config['channels'],
                    n_blocks=self.config['n_blocks'],
                    block_types=self.config['block_types'],
                    n_fmap=self.config['n_fmap'])

        gen = pgg.build_gen_stable()
        version = '{}_{}'.format(self.config['block'], self.config['steps'])
        load_weights(gen, 'gen', version, self.session_id)

        print('--Stable generator loaded')

        latents = random_latents(pgg.latent_size, n_latents)

        print('Generating images...')

        raw_images = tf.constant(gen.predict(latents, batch_size=16, verbose=1), dtype=tf.float32)
        images = combine_channels(raw_images)

        print('Scoring images...')

        scores = np.zeros(shape=[n_latents, 2])
        for i in range(n_latents):
            scores[i, 0] = K.mean(images[i])
            scores[i, 1] = K.std(images[i])
            if i % 1000 == 0:
                print('Progress: {}'.format(i / n_latents * 100))

        semantics = ['mean', 'std']

        for i in range(len(semantics)):
            self.create_boundary(semantics[i], latents, scores[:, i])

    def save(self):
        np.savez_compressed(self.path, **self.boundaries)
        print('--Boundaries saved')

    def move_latent(self, latent, semantic, alpha):
        return latent + alpha * self.boundaries[semantic]

    def move_latent_conditional(self):
        pass
