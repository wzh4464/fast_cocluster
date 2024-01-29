/**
 * File: /src/union_dc.rs
 * Created Date: Thursday, November 23rd 2023
 * Author: Zihan
 * -----
 * Last Modified: Tuesday, 16th January 2024 4:24:04 pm
 * Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 * -----
 * HISTORY:
 * Date      		By   	Comments
 * ----------		------	---------------------------------------------------------
 */
use std::collections::VecDeque;

pub struct UnionDC {}

trait UnionDCImpl<T> {
    fn union_dc(
        queue: &mut VecDeque<T>,
        func: impl Fn(T, T) -> T,
        end_condition: impl Fn(&VecDeque<T>) -> bool,
    );
}

impl UnionDC {
    pub fn union_dc<T, F>(
        queue: &mut VecDeque<T>,
        mut func: F,
        end_condition: impl Fn(&VecDeque<T>) -> bool,
    ) where
        F: FnMut(T, T) -> T,
    {
        while !end_condition(queue) {
            let a = queue
                .pop_front()
                .expect("Queue should have enough elements");
            let b = queue
                .pop_front()
                .expect("Queue should have enough elements");
            let result = func(a, b);
            queue.push_back(result);
        }
        // queue
    }
}
