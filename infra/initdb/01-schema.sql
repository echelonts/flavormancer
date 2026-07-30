-- Flavormancer schema (#20).
--
-- The substitution index is a nearest-neighbour search over the flavour-profile vector, so the
-- vector lives in the database rather than being recomputed per query. 177 dimensions:
-- 6 taste + 166 aroma + 5 mouthfeel. Tox is deliberately NOT in the vector — safety is not a
-- flavour-match dimension, and letting it steer "what tastes similar" would be wrong.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS molecule (
    inchikey_skel   text PRIMARY KEY,        -- stereo-agnostic connectivity skeleton
    smiles          text NOT NULL,
    name            text,                    -- never null in practice: falls back to formula
    mw              real,
    logp            real,
    tpsa            real,
    food_listed     boolean DEFAULT false,   -- open-gov register listing, NOT a safety clearance
    taste_documented text
);

CREATE TABLE IF NOT EXISTS molecule_profile (
    inchikey_skel   text PRIMARY KEY REFERENCES molecule(inchikey_skel) ON DELETE CASCADE,
    profile         vector(177) NOT NULL,
    aromas          text[]                   -- heads clearing their own calibrated threshold
);

-- Cosine distance: the profile is a direction in flavour space, and two molecules with the same
-- balance of notes at different intensities should still read as neighbours.
CREATE INDEX IF NOT EXISTS molecule_profile_cos
    ON molecule_profile USING hnsw (profile vector_cosine_ops);

CREATE INDEX IF NOT EXISTS molecule_food_listed ON molecule (food_listed);
