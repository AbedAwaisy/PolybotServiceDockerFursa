rsconf = {
    _id: "myReplicaSet",
    members: [
        { _id: 0, host: "mongo1:27017" },
        { _id: 1, host: "mongo2:27018" },
        { _id: 2, host: "mongo3:27019" }
    ]
};

rs.initiate(rsconf);