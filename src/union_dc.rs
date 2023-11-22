/*
 * File: /union_dc.rs
 * Created Date: Wednesday November 22nd 2023
 * Author: Zihan
 * -----
 * Last Modified: Wednesday, 22nd November 2023 11:26:45 pm
 * Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 * -----
 * HISTORY:
 * Date      		By   	Comments
 * ----------		------	---------------------------------------------------------
 */

use std::collections::VecDeque;

pub struct UnionDC {}

impl UnionDC {
    pub fn union_dc<T, F>(
        queue: &mut VecDeque<T>,
        mut func: F,
        end_condition: impl Fn(&VecDeque<T>) -> bool,
    )
    where
        F: FnMut(T, T) -> T,
    {
        let mut in_or_not = end_condition(queue);
        while !in_or_not {
            let a = queue
                .pop_front()
                .expect("Queue should have enough elements");
            let b = queue
                .pop_front()
                .expect("Queue should have enough elements");
            let result = func(a, b);
            queue.push_back(result);
            in_or_not = end_condition(queue);
        }
        // queue
    }
}
