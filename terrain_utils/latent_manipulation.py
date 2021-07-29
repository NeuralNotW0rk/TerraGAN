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

    def create_boundary(self, semantic, latents, scores, percentile=0.5):

        # Sort data wrt scores
        idx_s = np.argsort(scores)
        latents_sorted = latents[idx_s]
        scores_sorted = scores[idx_s]

        # Split scores across origin
        split = int(len(scores_sorted) * percentile)
        scores_sorted -= scores_sorted[split] + 1e-8
        labels = np.sign(scores_sorted)

        # Find boundary
        print('Fitting new boundary for', semantic)
        model = svm.SVC(kernel='linear')
        model.fit(latents_sorted, labels)
        boundary = model.coef_.reshape(1, latents_sorted.shape[1]).astype(np.float32)
        boundary / np.linalg.norm(boundary)

        self.boundaries[semantic] = boundary
        print('--Boundary added')

    def process_scores(self, n_latents=10000, n_sections=10, from_file=False):

        self.boundaries = {}

        if from_file:

            # Load scores from file
            data_path = root_dir + 'data/' + self.session_id + '_scores_' + '_'.join(self.semantics) + '.npz'
            data = np.load(data_path)
            latents = data['latents']
            scores = data['scores']

        else:

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
            norm = (raw_images[:, :, :, 0] + 1.0) / 2.0
            images = (raw_images[:, :, :, 1] + 1.0) / 2.0

            print('Scoring images...')

            scores = np.zeros(shape=[n_latents, 3])
            for i in range(n_latents):
                scores[i, 0] = K.mean(images[i])
                scores[i, 1] = K.std(images[i])
                scores[i, 2] = K.max(images[i])
                if i % 1000 == 0:
                    print('Progress: {}'.format(i / n_latents * 100))

            self.semantics = ['mean', 'std', 'max']

        for i in range(len(self.semantics)):
            for j in range(1, n_sections):
                p = j / n_sections
                self.create_boundary(self.semantics[i] + '_' + str(j), latents, scores[:, i], percentile=p)

    def save(self):
        np.savez_compressed(self.path, **self.boundaries)
        print('--Boundaries saved')

    def center_latent(self, latent, semantic):
        n = self.boundaries[semantic][0]
        z = latent
        z -= n
        n /= np.linalg.norm(n)
        z = z - n * np.dot(n[0], z)
        return z

    def move_latent(self, latent, semantic, delta):
        return latent + delta * self.boundaries[semantic][0]

    def move_latent_conditional(self):
        pass
