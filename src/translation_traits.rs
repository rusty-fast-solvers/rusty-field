//! Traits for the translation of fields

pub trait FmmTranslation {

    // Map particles to a local expansion.
    // # Arguments
    // * `particles`: A (3, N) array of particles.
    // * `indices`: Column indices of the particles to be added to the multipole expansion.
    fn p2m(&mut self, index: usize);

    // Propagate a multipole expansion to the parent.
    fn m2m(&mut self, index: usize);

    // Update a local expansion from a multipole expansion.
    fn m2l(&mut self, local: usize, other: usize);

    // Update a local expansion from the parent
    fn l2l(&mut self, index: usize);

    // Evaluate a local expansion at particle locations.
    fn l2p(&mut self, index: usize);

    // Direct evaluation of particle interactions between two boxes.
    fn p2p(&mut self, index: usize, other: usize);
}

