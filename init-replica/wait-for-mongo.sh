#!/bin/bash
set -e

host="$1"
shift
cmd="$@"

until mongo --host "$host" --eval 'quit(db.runCommand({ ping: 1 }).ok ? 0 : 2)'; do
  >&2 echo "Mongo is unavailable - sleeping"
  sleep 5
done

>&2 echo "Mongo is up - executing command"
exec $cmd
