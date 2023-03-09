#4. DSSM实现文本相似度
def get_model(embedding_matrix, nb_words, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, num_lstm, rate_drop_dense):
 
 
    att1_layer = Attention.Attention(MAX_SEQUENCE_LENGTH)
 
    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')   # 编码后的问题1的词特征
    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')   # 编码后的问题2的词特征
 
    # embedding
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    embedded_sequences_2 = embedding_layer(sequence_2_input)
 
    # encode
    lstm1_layer = Bidirectional(LSTM(num_lstm))
    encode_sequences_1 = lstm1_layer(embedded_sequences_1)
    encode_sequences_2 = lstm1_layer(embedded_sequences_2)
 
    # lstm
    lstm0_layer = LSTM(num_lstm, return_sequences=True)
    lstm2_layer = LSTM(num_lstm)
    v1ls = lstm2_layer(lstm0_layer(embedded_sequences_1))
    v2ls = lstm2_layer(lstm0_layer(embedded_sequences_2))
    v1 = Concatenate(axis=1)([att1_layer(embedded_sequences_1), encode_sequences_1])
    v2 = Concatenate(axis=1)([att1_layer(embedded_sequences_2), encode_sequences_2])
 
    # sequence_1c_input = Input(shape=(MAX_SEQUENCE_LENGTH_CHAR,), dtype='int32')  # 编码后的问题1的字特征
    # sequence_2c_input = Input(shape=(MAX_SEQUENCE_LENGTH_CHAR,), dtype='int32')  # 编码后的问题2的字特征
 
    # embedding_char_layer = Embedding(char_words,
    #                             EMBEDDING_DIM)
 
    # embedded_sequences_1c = embedding_char_layer(sequence_1c_input)
    # embedded_sequences_2c = embedding_char_layer(sequence_2c_input)
 
    # x1c = lstm1_layer(embedded_sequences_1c)
    # x2c = lstm1_layer(embedded_sequences_2c)
    # v1c = Concatenate(axis=1)([att1_layer(embedded_sequences_1c), x1c])
    # v2c = Concatenate(axis=1)([att1_layer(embedded_sequences_2c), x2c])
 
    # compose
    mul = Multiply()([v1, v2])
    sub = Lambda(lambda x: K.abs(x))(Subtract()([v1, v2]))
    maximum = Maximum()([Multiply()([v1, v1]), Multiply()([v2, v2])])
    # mulc = Multiply()([v1c, v2c])
    # subc = Lambda(lambda x: K.abs(x))(Subtract()([v1c, v2c]))
    # maximumc = Maximum()([Multiply()([v1c, v1c]), Multiply()([v2c, v2c])])
    sub2 = Lambda(lambda x: K.abs(x))(Subtract()([v1ls, v2ls]))
    # matchlist = Concatenate(axis=1)([mul, sub, mulc, subc, maximum, maximumc, sub2])
    matchlist = Concatenate(axis=1)([mul, sub, maximum, sub2])
    matchlist = Dropout(rate_drop_dense)(matchlist)
 
    matchlist = Concatenate(axis=1)(
        [Dense(32, activation='relu')(matchlist), Dense(48, activation='sigmoid')(matchlist)])
    res = Dense(1, activation='sigmoid')(matchlist)
 
    # model = Model(inputs=[sequence_1_input, sequence_2_input,
    #                       sequence_1c_input, sequence_2c_input], outputs=res)
    model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=res)
    model.compile(optimizer=Adam(lr=0.001), loss="binary_crossentropy", metrics=['acc'])
    model.summary()
    return model